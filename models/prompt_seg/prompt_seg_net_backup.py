import torch
import torch.nn as nn
from utils import gene_max_area_box,one_hot_encoder,get_prompt
from models.sam.build_sam import sam_model_registry
from .mask_decoder import MaskDecoder
from .cls_transformer import TwoWayTransformer
import torch.nn.functional as F
import numpy as np
import cv2

class PromptSegNet(nn.Module):

    def __init__(self, 
                 num_classes = 1,
                 sm_depth = 1,
                 use_inner_feat = False,
                 use_multi_mlps = False,
                 use_mask_prompt = False,
                 update_prompt_encoder = False,
                 use_embed = False,
                 sam_ckpt = None,
                 device = None,
                 ignore_idx = 255,
                ):
        super(PromptSegNet, self).__init__()

        self.use_multi_mlps = use_multi_mlps
        self.use_mask_prompt = use_mask_prompt
        self.use_inner_feat = use_inner_feat
        self.update_prompt_encoder = update_prompt_encoder
        self.use_embed = use_embed
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.device = device
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
            use_multi_mlps = use_multi_mlps,
        )

        self.except_keys = [
            'cls_token', 
            'cls_prediction_head',
            'output_tokens',
            'output_mlps',
            'process_inter_feat',
            'transformer.layers.0.semantic_module',
            'transformer.layers.1.semantic_module',
        ]

        self.load_sam_parameters(sam.mask_decoder.state_dict())
        self.freeze_parameters()

    def load_sam_parameters(self, sam_mask_decoder_params: dict):
        decoder_state_dict = {}
        discard_keys = [
            'output_hypernetworks_mlps',
            'iou_token',
            'mask_tokens',
            'iou_prediction_head'
        ]
        for key,value in sam_mask_decoder_params.items():
            key_first = key.split('.')[0]
            if key_first not in [*self.except_keys, *discard_keys]:
                decoder_state_dict[key] = value

        print('='*10 + ' load parameters from sam ' + '='*10)
        print(self.mask_decoder.load_state_dict(decoder_state_dict, strict = False))
        use_weight = sam_mask_decoder_params['mask_tokens.weight'][2].unsqueeze(0)
        repeat_num = self.num_classes if self.use_multi_mlps else 1
        cls_token_init_weight = torch.repeat_interleave(use_weight, repeat_num, dim=0)
        self.mask_decoder.output_tokens.weight = nn.Parameter(cls_token_init_weight)
        print('='*59)

    def freeze_parameters(self):
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.prompt_encoder.named_parameters():
            param.requires_grad = self.update_prompt_encoder
        
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

    def forward_single_class(self, sampled_batch, prompt_type):
        mask_1024 = sampled_batch['mask_1024'] # shape: [bs, 1024, 1024]
        bs_image_embedding,inter_feature = self.gene_img_embed(sampled_batch)

        type_to_repeat = dict(
            max_bbox = 2,
            random_bbox = 2,
            random_point = 2,
            max_bbox_center_point = 2,
            max_bbox_with_point = 3,
            all_bboxes = 0
        )
        bs_sparse_embedding = []
        bs_prompt = []
        bs = mask_1024.shape[0]
        for idx in range(bs):
            boxes, points = None, None
            if prompt_type is not None:
                if 'gt_boxes' in sampled_batch.keys():
                    all_gtboxes = []
                    gt_boxes = sampled_batch['gt_boxes'][idx]
                    coord_ratio = sampled_batch['coord_ratio'][idx].item()
                    for cls_id, boxes in gt_boxes.items():
                        all_gtboxes.extend(boxes)

                    boxes, points, _ = get_prompt(prompt_type, None, np.array(all_gtboxes), self.device, coord_ratio)
                else:
                    single_label = mask_1024[idx]
                    boxes, points, _ = get_prompt(prompt_type, single_label.numpy(), None, self.device, 1)
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points = points,
                boxes = boxes,
                masks = None,
            )
            if sparse_embeddings.shape[1] == 0 and prompt_type is not None:
                repeat_num = type_to_repeat[prompt_type]
                sparse_embeddings = self.prompt_encoder.not_a_point_embed.weight.unsqueeze(0)
                sparse_embeddings = torch.repeat_interleave(sparse_embeddings, repeat_num, dim=1)
            bs_sparse_embedding.append(sparse_embeddings)
        
            bs_prompt.append(dict(
                points = points,
                boxes = boxes,
            ))
        bs_sparse_embedding = torch.cat(bs_sparse_embedding, dim=0)
        
        image_pe = self.prompt_encoder.get_dense_pe()
        # low_res_masks.shape: (bs, num_cls, 256, 256)
        low_res_masks = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = bs_sparse_embedding,
            dense_prompt_embeddings = dense_embeddings,
            inter_feature = inter_feature
        )
        logits_512 = F.interpolate(
            low_res_masks,
            (512, 512),
            mode="bilinear",
            align_corners=False,
        )
        outputs = {
            'logits_256': low_res_masks,
            'logits_512': logits_512,
            'prompts': bs_prompt,
        }
        return outputs
    
    def forward_multi_class(self, sampled_batch, prompt_type, mask_256_logits=None):
        mask_512 = sampled_batch['mask_512'].to(self.device)
        gt_boxes,coord_ratio = None,1
        # one_hot_gt_mask_512.shape: (1, num_cls, h, w)
        one_hot_gt_mask_512 = one_hot_encoder(self.num_classes, mask_512)

        boxes,points,masks = None, None, None
        # 若当前图像没有任何前景包围框，则它应该输出全零图
        target_cls = (torch.ones((1,), dtype=torch.uint8) * self.ignore_idx).to(self.device)
        target_masks = torch.zeros_like(mask_512).to(self.device)

        all_gtboxes, all_gtboxes_clsid, all_gtboxes_gtmasks = [],[],[]
        if 'gt_boxes' in sampled_batch.keys():
            gt_boxes = sampled_batch['gt_boxes'][0]
            coord_ratio = sampled_batch['coord_ratio'][0].item()
            for cls_id, boxes in gt_boxes.items():
                cls_id = int(cls_id)
                choice_boxes, _, sample_idx = get_prompt(prompt_type, None, np.array(boxes), self.device, coord_ratio)
                all_gtboxes.extend(choice_boxes)
                all_gtboxes_clsid.extend([cls_id]*len(choice_boxes))
                gtmasks = torch.repeat_interleave(one_hot_gt_mask_512[:,cls_id, ::], len(choice_boxes), dim=0)
                all_gtboxes_gtmasks.extend(gtmasks)
            
        if len(all_gtboxes) > 0:
            target_cls = torch.as_tensor(all_gtboxes_clsid, device=self.device)
            target_masks = torch.stack(all_gtboxes_gtmasks).to(self.device)
            boxes = torch.stack(all_gtboxes).to(self.device)
            if self.use_mask_prompt:
                k_boxes = boxes.shape[0]
                masks = torch.repeat_interleave(mask_256_logits, k_boxes, dim=0).float().unsqueeze(1)

        bs_image_embedding,inter_feature = self.gene_img_embed(sampled_batch)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points = points,
            boxes = boxes,
            masks = masks,   # masks.shape: (k_num_boxes or 1, 1, 256, 256)
        )
        image_pe = self.prompt_encoder.get_dense_pe()
        # low_logits.shape: (k_num_boxes or 1, num_cls or 1, 256, 256)
        # cls_logits.shape: (k_num_boxes or 1, num_cls)
        low_logits,cls_logits = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings = dense_embeddings,
            inter_feature = inter_feature
        )
        logits_512 = F.interpolate(
            low_logits,
            (512, 512),
            mode="bilinear",
            align_corners=False,
        )
        if self.use_multi_mlps:
            # cls_pred.shape: (k_num_boxes or 1, )
            cls_pred = torch.argmax(cls_logits, dim=1)
            cls_low_logits = [logits_512[i, cls_pred[i], ::] for i in range(len(cls_pred))]
            cls_low_logits = torch.stack(cls_low_logits,dim=0).unsqueeze(1)
        else:
            cls_low_logits = logits_512

        outputs = dict(
            pred_mask_logits = cls_low_logits, target_masks = target_masks,
            pred_cls_logits = cls_logits, target_cls = target_cls,
            prompts = dict(
                points = points,
                boxes = boxes,
            )
        )
    
        return outputs

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
    
    