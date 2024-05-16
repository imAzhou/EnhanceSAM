import torch
import torch.nn as nn
from utils import gene_point_embed
from models.sam.build_sam import sam_model_registry
from .mask_decoder import MaskDecoder
from .cls_transformer import TwoWayTransformer
import torch.nn.functional as F
from utils.tools import gene_max_area_box
import numpy as np

class ClsProposalNet(nn.Module):

    def __init__(self, 
                 num_points = [0,0],
                 num_classes = 1,
                 useModule = None,
                 sam_ckpt = 'checkpoints_sam/sam_vit_h_4b8939.pth'
                ):
        super(ClsProposalNet, self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        sam = sam_model_registry['vit_h'](checkpoint = sam_ckpt)
        self.prompt_encoder = sam.prompt_encoder
        transformer_dim = sam.mask_decoder.transformer_dim
        self.mask_decoder = MaskDecoder(
            num_classes_outputs = num_classes,
            transformer = TwoWayTransformer(
                depth=2,
                embedding_dim = transformer_dim,
                mlp_dim = 2048,
                num_heads = 8,
                useModule = useModule,
            ),
            transformer_dim = transformer_dim
        )

        self.except_keys = [
            'cls_tokens', 
            'cls_mlps',
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

    def _gene_sparse_embed(self, mask_np, device):
        points,boxes = None, None
        center_point,bbox = None, None
        if np.sum(mask_np) > 0:
            center_point,bbox = gene_max_area_box(mask_np)
            coords_torch = torch.as_tensor(np.array([center_point]), dtype=torch.float, device=device)
            labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            points = (coords_torch, labels_torch)

            box_np = np.array(bbox)
            box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)
            boxes = box_torch[None, :]

        sparse_embeddings, _ = self.prompt_encoder(
            points = points,
            boxes = boxes,
            masks = None,
        )
        return sparse_embeddings, (center_point,bbox)

    def forward(self, bs_image_embedding: torch.Tensor, label_mask: torch.Tensor):
        device = bs_image_embedding.device
        bs_point_box = []

        bs_sparse_embedding = []
        for single_label in label_mask:
            # sparse_embedding,single_img_points = gene_point_embed(self, single_label.cpu(), self.num_points, device)
            sparse_embedding, (point,box) = self._gene_sparse_embed(single_label.cpu().numpy(),device)
            # there is no positive region
            if sparse_embedding.shape[1] == 0:
                sparse_embedding = self.prompt_encoder.not_a_point_embed.weight.unsqueeze(0)
                # when prompt box is none, prompt encoder will concat more point embedding
                sparse_embedding = torch.repeat_interleave(sparse_embedding, 3, dim=1)
            
            bs_point_box.append({
                'point': point,
                'box': box,
            })
            bs_sparse_embedding.append(sparse_embedding)
            
        bs_sparse_embedding = torch.cat(bs_sparse_embedding, dim=0)
        
        b,c,h,w = bs_image_embedding.shape
        p = self.prompt_encoder
        dense_embeddings = p.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            b, -1, h, w
        )

        image_pe=self.prompt_encoder.get_dense_pe()
        low_res_masks = self.mask_decoder(
            image_embeddings=bs_image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=bs_sparse_embedding,
            dense_prompt_embeddings=dense_embeddings,
        )
        input_sam_size_masks = F.interpolate(
            low_res_masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        outputs = {
            'pred_mask_512': low_res_masks,
            'pred_mask_1024': input_sam_size_masks,
            'bs_point_box': bs_point_box
        }

        return outputs
    
    def forward_with_ppo(self, bs_image_embedding: torch.Tensor,
                input_boxes: torch.Tensor):
        '''
        Args:
            bs_image_embedding: tensor, shape is (bs, 256, 64, 64)
            input_boxes: tensor, shape is (bs, num_boxes, 4), coord format is x1y1x2y2 in 1024*1024 scale
        '''
        bs_low_res_masks = []
        for single_boxes,single_emb in zip(input_boxes,bs_image_embedding):
            if torch.sum(single_boxes) < 0:
                single_boxes = None
            # single_boxes.shape: (num_boxes, 4)
            # sparse_embeddings.shape: (num_boxes, 2, 256)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=single_boxes,
                masks=None,
            )
            image_pe=self.prompt_encoder.get_dense_pe()
            # low_res_masks.shape: (num_boxes, 1, 256, 256)
            low_res_masks = self.mask_decoder(
                image_embeddings=single_emb.unsqueeze(0),
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            if sparse_embeddings.shape[1] == 0:
                low_res_masks = torch.repeat_interleave(low_res_masks, self.num_points[0], dim=0)
            
            low_res_masks = torch.mean(low_res_masks, dim=0)
            bs_low_res_masks.append(low_res_masks)
        
        bs_low_res_masks = torch.cat(bs_low_res_masks, dim=0)
        low_res_h,low_res_w = bs_low_res_masks.shape[-2:]
        bs_low_res_masks = bs_low_res_masks.view((-1, 1, low_res_h,low_res_w)).contiguous()
        
        # input_sam_size_masks.shape: (bs, num_classes, 1024, 1024)
        input_sam_size_masks = F.interpolate(
            bs_low_res_masks,
            (self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        outputs = {
            'low_res_masks': bs_low_res_masks,
            'pred_logits': input_sam_size_masks,
        }

        return outputs
