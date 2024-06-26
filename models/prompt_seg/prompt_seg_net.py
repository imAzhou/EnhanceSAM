import torch
import torch.nn as nn
from utils import gene_point_embed,gene_max_area_box,gene_bbox_for_mask
from models.sam.build_sam import sam_model_registry
from .mask_decoder import MaskDecoder
# from .cls_transformer import TwoWayTransformer
from .cls_transformer_backup import TwoWayTransformer
import torch.nn.functional as F
import numpy as np

class PromptSegNet(nn.Module):

    def __init__(self, 
                 num_classes = 1,
                 sm_depth = 1,
                 use_inner_feat = False,
                 use_embed = False,
                 sam_ckpt = None,
                 device = None,
                 prompt_types = []
                ):
        super(PromptSegNet, self).__init__()

        assert all(item in ['box','point','all_boxes'] for item in prompt_types), \
        "One or more prompt type are not allowed."

        self.use_inner_feat = use_inner_feat
        self.use_embed = use_embed
        self.num_classes = num_classes
        self.device = device
        self.prompt_types = prompt_types
        sam = sam_model_registry['vit_h'](checkpoint = sam_ckpt)
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.preprocess = sam.preprocess
        
        transformer_dim = sam.mask_decoder.transformer_dim
        self.mask_decoder = MaskDecoder(
            num_classes_outputs = num_classes,
            transformer = TwoWayTransformer(
                depth=2,
                embedding_dim = transformer_dim,
                mlp_dim = 2048,
                num_heads = 8,
                sm_depth = sm_depth,
            ),
            transformer_dim = transformer_dim,
            encoder_embed_dim = 1280,
            use_inner_feat = use_inner_feat,
        )

        self.except_keys = [
            'cls_tokens', 
            'cls_mlps',
            'process_inter_feat',
            'upscaling_inter_feat',
            'transformer.layers.0.semantic_module',
            'transformer.layers.1.semantic_module',
        ]

        self.load_sam_parameters(sam.mask_decoder.state_dict())
        self.freeze_parameters()

    def load_sam_parameters(self, sam_mask_decoder_params: dict):
        decoder_state_dict = {}
        discard_keys = [
            'output_hypernetworks_mlps',
        ]
        for key,value in sam_mask_decoder_params.items():
            key_first = key.split('.')[0]
            if key_first not in [*self.except_keys, *discard_keys]:
                decoder_state_dict[key] = value

        print('='*10 + ' load parameters from sam ' + '='*10)
        print(self.mask_decoder.load_state_dict(decoder_state_dict, strict = False))
        use_weight = sam_mask_decoder_params['mask_tokens.weight'][2].unsqueeze(0)
        cls_token_init_weight = torch.repeat_interleave(use_weight, self.num_classes, dim=0)
        self.mask_decoder.cls_tokens.weight = nn.Parameter(cls_token_init_weight)
        print('='*59)

    def freeze_parameters(self):
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.prompt_encoder.named_parameters():
            param.requires_grad = False
        
        self.except_keys.append('output_upscaling')
        # freeze transformer
        for name, param in self.mask_decoder.named_parameters():
            need_update = False
            for key in self.except_keys:
                if key in name:
                    need_update = True
                    break
            param.requires_grad = need_update
            

    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        merged_dict = {
            'mask_decoder': self.mask_decoder.state_dict()
        }
        torch.save(merged_dict, filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)
        print('='*10 + ' load parameters for mask decoder ' + '='*10)
        print(self.mask_decoder.load_state_dict(state_dict['mask_decoder'], strict = False))
        print('='*59)

    def forward(self, sampled_batch: dict):
        mask_1024 = sampled_batch['mask_1024']  # shape: (bs,h,w)
        bs_image_embedding,inter_feature = self.gene_img_embed(sampled_batch)

        all_cls_logits,all_cls_prompts = [],[]
        for cls_i in range(self.num_classes):
            mask_1024_cls_i = (mask_1024 == (1 if self.num_classes == 1 else cls_i)).to(torch.uint8)
            low_logits, prompts = self.forward_single_class(bs_image_embedding,inter_feature, mask_1024_cls_i)
            low_logits_cls_i = low_logits[:,cls_i,::].unsqueeze(1)
            all_cls_logits.append(low_logits_cls_i)
            all_cls_prompts.append(prompts)

        # all_cls_logits.shape: (bs, num_class, h ,w)
        all_cls_logits = torch.cat(all_cls_logits, dim=1)
        all_cls_logits_1024 = F.interpolate(
            all_cls_logits,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        outputs = {
            'pred_mask_512': all_cls_logits,
            'pred_mask_1024': all_cls_logits_1024,
            'bs_point_box': all_cls_prompts,    # [[{point:[[x1,y1],[x1,y1]], box:[x1,y1,x2,y2]},{}],[],...,[]]
        }

        return outputs

    def forward_single_class(self, bs_image_embedding, inter_feature, mask_1024_cls_i):
        sparse,dense,prompts = self.gene_prompt_embed(bs_image_embedding, mask_1024_cls_i)
        image_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse,
            dense_prompt_embeddings = dense,
            inter_feature = inter_feature
        )
        return low_res_masks, prompts

    def gene_prompt_embed(self, bs_image_embedding, label_mask):
        bs_sparse_embedding = []
        bs_point_box = []
        point,box,all_boxes = None,None,None
        for single_label in label_mask:
            if 'all_boxes' in self.prompt_types:    # batch_size always equal 1
                # sparse_embeddings.shape: (k_boxes, 2, 256) or (1, 0, 256)
                sparse_embedding, all_boxes = self._gene_sparse_embed_by_allbox(single_label.numpy(),self.device)
            else:
                # sparse_embeddings.shape: (bs, 3, 256) or (1, 0, 256)
                sparse_embedding, (point,box) = self._gene_sparse_embed(single_label.numpy(),self.device)
                # there is no positive point
                if sparse_embedding.shape[1] == 0 and len(self.prompt_types) > 0:
                    sparse_embedding = self.prompt_encoder.not_a_point_embed.weight.unsqueeze(0)
                    # when prompt box is none, prompt encoder will concat more point embedding
                    repeat_time = 3 if len(self.prompt_types) == 2 else 2
                    sparse_embedding = torch.repeat_interleave(sparse_embedding, repeat_time, dim=1)

            bs_point_box.append({
                'point': point,
                'box': box,
                'all_boxes': all_boxes
            })
            bs_sparse_embedding.append(sparse_embedding)

        bs_sparse_embedding = torch.cat(bs_sparse_embedding, dim=0)
        _,c,h,w = bs_image_embedding.shape
        p = self.prompt_encoder
        dense_embeddings = p.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs_sparse_embedding.shape[0], -1, h, w
        )

        return bs_sparse_embedding, dense_embeddings, bs_point_box

    def gene_img_embed(self, sampled_batch: dict):
        if self.use_embed:
            bs_image_embedding = sampled_batch['img_embed'].to(self.device)
            inter_feature = sampled_batch['img_embed_inner'].to(self.device) if self.use_inner_feat else None
        else:
            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                input_images = self.preprocess(sampled_batch['input_image']).to(self.device)
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                if self.use_inner_feat:
                    bs_image_embedding,inter_feature = self.image_encoder(input_images, need_inter=True)
                    bs_image_embedding,inter_feature = bs_image_embedding.detach(),inter_feature.detach()
                else:
                    bs_image_embedding = self.image_encoder(input_images, need_inter=False).detach()
                    inter_feature = None

        return bs_image_embedding,inter_feature
    
    def _gene_sparse_embed(self, mask_np, device):
        points,boxes = None, None
        center_point,bbox = None, None
        if np.sum(mask_np) > 0:
            center_point,bbox = gene_max_area_box(mask_np)
            coords_torch = torch.as_tensor(np.array([center_point]), dtype=torch.float, device=device)
            labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            points = (coords_torch, labels_torch)

            box_np = np.array(bbox) # [x1,y1,x2,y2]
            box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
            boxes = box_torch[None, :]  #[[[x1,y1,x2,y2]]]

        if 'box' not in self.prompt_types:
            boxes = None
            bbox = None
        if 'point' not in self.prompt_types:
            points = None
            center_point = None

        sparse_embeddings, _ = self.prompt_encoder(
            points = points,
            boxes = boxes,
            masks = None,
        )

        return sparse_embeddings, (center_point,bbox)
    
    def _gene_sparse_embed_by_allbox(self, mask_np, device):
        points,boxes = None, None
        all_boxes = []
        if np.sum(mask_np) > 0:
            all_boxes = gene_bbox_for_mask(mask_np)
            box_np = np.array(all_boxes)
            boxes = torch.as_tensor(box_np, dtype=torch.float, device=device)
            boxes = boxes.unsqueeze(1)  # (k_box_nums, 1, 4)
        
        # sparse_embeddings.shape: (k_boxes, 2, 256) or (1, 0, 256)
        sparse_embeddings, _ = self.prompt_encoder(
            points = points,
            boxes = boxes,
            masks = None,
        )
        return sparse_embeddings, all_boxes
    