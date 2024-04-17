import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sam import Sam
from typing import List,Tuple
from .mask_decoder import MaskDecoder
from .mask_transformer import TwoWayTransformer
# from .extra_head_transformer import TwoWayTransformer
# from .task_head_3 import TaskHead
# from .task_head_2 import TaskHead
from .task_head_1 import TaskHead
from utils import SegDataSample
from help_func.tools import gene_point_embed
from mmengine.structures import PixelData


class MaskProposalNet(nn.Module):
    """
    Args:
        sam_model: a vision transformer model, see base_vit.py
        num_classes: how many classes the model output, default to the vit model
    """

    def __init__(self, 
                 sam_model: Sam, 
                 num_classes: int=None,
                 num_queries: int=None,
                #  point_num: int=1000,
                 sam_ckpt_path: str=None,
                 use_embed: bool=True,
                 conv_size: int=3,
                ):
        super(MaskProposalNet, self).__init__()
        self.sam_ckpt_path = sam_ckpt_path
        self.use_embed = use_embed
        # self.point_num = point_num

        self.input_size = sam_model.image_encoder.img_size
        self.preprocess = sam_model.preprocess
        self.postprocess_masks = sam_model.postprocess_masks
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        transformer_dim = sam_model.mask_decoder.transformer_dim
        self.mask_decoder = MaskDecoder(
            num_classes = num_classes,
            num_queries = num_queries,
            transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
                conv_size=conv_size
            ),
            transformer_dim = transformer_dim
        )
        self.task_head = TaskHead(
            num_classes = num_classes,
            num_queries = num_queries,
        )

        self.load_sam_parameters()
        self.freeze_parameters()

    def load_sam_parameters(self):
        # load decoder parameters
        self.except_keys = [
            'mask_tokens', 
            # 'output_hypernetworks_mlps', 
            'cls_embed',
            'mask_embed',
            'transformer.layers.0.cls_conv',
            'transformer.layers.1.cls_conv',
            # 'extra',
            # 'norm'
        ]
        sam_state_dict = torch.load(self.sam_ckpt_path)
        encoder_state_dict = {}
        promt_state_dict = {}
        decoder_state_dict = {}
        for key,value in sam_state_dict.items():
            new_key = key.split('.')[1:]
            new_key = '.'.join(new_key)
            if 'image_encoder' in key:
                encoder_state_dict[new_key] = value
            if 'prompt_encoder' in key:
                promt_state_dict[new_key] = value
            if 'mask_decoder' in key:
                key_first = new_key.split('.')[0]
                if key_first not in self.except_keys:
                    decoder_state_dict[new_key] = value

        print('='*10 + ' load parameters from sam ' + '='*10)
        self.image_encoder.load_state_dict(encoder_state_dict)
        self.prompt_encoder.load_state_dict(promt_state_dict)
        print(self.mask_decoder.load_state_dict(decoder_state_dict, strict = False))
        print('='*59)

    def freeze_parameters(self):
        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
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
        self.mask_decoder.load_state_dict(state_dict['mask_decoder'])

    def forward(self, bs_image_embedding: torch.Tensor, gt_label_batch: torch.Tensor):
        
        # device = bs_image_embedding.device
        # b,c,h,w = bs_image_embedding.shape
        # gt_image_mask = gt_label_batch != 255
        # bs_sparse_embedding: List[torch.Tensor] = []
        # for label_mask in gt_image_mask:
        #     sparse_embedding,_ = gene_point_embed(self, label_mask.cpu(), (self.point_num, 0), device)
        #     # there is no positive point
        #     if sparse_embedding.shape[1] == 0:
        #         sparse_embedding = self.prompt_encoder.not_a_point_embed.weight.unsqueeze(0)
        #         # when prompt box is none, prompt encoder will concat more point embedding
        #         sparse_embedding = torch.repeat_interleave(sparse_embedding, sum(self.point_num)+1, dim=1)
            
        #     bs_sparse_embedding.append(sparse_embedding)
        # bs_sparse_embedding = torch.cat(bs_sparse_embedding, dim=0)
        # dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        #     b, -1, h, w
        # )
             
        image_pe=self.prompt_encoder.get_dense_pe()
        bs_sparse_embedding, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        
        cls_pred, masks_pred = self.mask_decoder(
            image_embeddings=bs_image_embedding,
            # image_pe: (bs_img, [C]256, [H]64, [W]64)
            image_pe=image_pe,
            sparse_prompt_embeddings=bs_sparse_embedding,
            dense_prompt_embeddings=dense_embeddings,
        )
        # input_sam_size_masks = F.interpolate(
        #     masks_pred,
        #     (self.input_size, self.input_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # outputs = {
        #     'pred_logits': input_sam_size_masks
        # }

        return cls_pred, masks_pred

    def loss(self, 
             bs_image_embedding: torch.Tensor,
             gt_label_batch: torch.Tensor
        ):

        data_samples = [
            SegDataSample(gt_sem_seg=PixelData(data = gt_label_mask.long()))
            for gt_label_mask in gt_label_batch
        ]

        cls_pred, masks_pred = self(bs_image_embedding, gt_label_batch)
        loss_dict = self.task_head(cls_pred, masks_pred, gt_label_batch)
        # loss_dict = self.task_head(cls_pred, masks_pred, data_samples)
        return loss_dict

    def predict(self, bs_image_embedding: torch.Tensor, gt_label_batch: torch.Tensor):


        cls_pred, masks_pred = self(bs_image_embedding, gt_label_batch)
        # upsample mask
        mask_pred_results = F.interpolate(
            masks_pred, 
            size=self.input_size, 
            mode='bilinear', 
            align_corners=False)
        # cls_score = F.softmax(cls_pred, dim=-1)[..., :-1]
        cls_score = F.softmax(cls_pred, dim=-1)
        mask_pred = mask_pred_results.sigmoid()
        # seg_logits.shape: (bs, num_class, h, w)
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        outputs = {
            'pred_logits': seg_logits
        }
        return outputs


        
