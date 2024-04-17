import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.utils import Upsample
from models.sam import LayerNorm2d
from utils.sample_point_calc_embed import sample_points_from_mask, calculate_point_embed

class DiscriminatorNet(nn.Module):
    def __init__(self, emb_dim, sample_points):
      super(DiscriminatorNet, self).__init__()
      
      concat_dim = 16
      self.num_classes = 1
      self.sample_points = sample_points
      self.cls_tokens = nn.Embedding(self.num_classes, emb_dim)
      self.cls_token_mlp = nn.Sequential(
        nn.Linear(emb_dim, concat_dim),
        nn.ReLU(),
      )
      self.img_embeding_upscaling = nn.Sequential(
          nn.ConvTranspose2d(emb_dim, emb_dim // 4, kernel_size=2, stride=2),
          LayerNorm2d(emb_dim // 4),
          nn.GELU(),
          ConvModule(
            in_channels=emb_dim // 4,
            out_channels=emb_dim // 8,
            kernel_size=3,
            padding='same',
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU')),
          Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False),
          ConvModule(
            in_channels=emb_dim // 8,
            out_channels=concat_dim,
            kernel_size=3,
            padding='same',
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU')),
          Upsample(
            scale_factor=4,
            mode='bilinear',
            align_corners=False)
      )
      
      self.finle_mlp = nn.Sequential(
        nn.Linear(concat_dim*2, concat_dim),
        nn.ReLU(),
        nn.Linear(concat_dim, 1)
      )
    
    def forward(self, 
                coarse_mask: torch.Tensor,
                sam_seg_mask: torch.Tensor,
                image_embedding: torch.Tensor
                ):
        '''
        Args:
          - coarse_mask (torch.Tensor): Binary tensor with shape (bs, h, w)
          - mask_index (torch.Tensor): Integer tensor with shape (bs, h, w)
        Return:
          - logits: the pred logits of point positivity with shape (bs, num_points)
          - sample_points: (bs, num_points, 2) 2-(y,x),  (y x) ∈ [0,1024]
        '''
        bs = image_embedding.size(0)
        # todo: 对 k 个points 加扰动，得到 k*5 个point，对扰动后的每组 point(1+4) 判别是否属于同一个sam_mask_index，
        # 是的话就认为采样的点在中心区域，而不是不确定的边缘区域
        # sample_points: (bs, num_points, 2) 2-(y,x), 1024 尺度上的点坐标
        sample_points, point_mask_indices = sample_points_from_mask(
                coarse_mask, sam_seg_mask, self.sample_points)
        
        upsample_img_embedding = self.img_embeding_upscaling(image_embedding)

        # point_embed: (bs, num_points, c).
        point_embed = calculate_point_embed(
                upsample_img_embedding, point_mask_indices, sam_seg_mask)
        bs_cls_tokens = self.cls_tokens.weight.unsqueeze(0).expand(bs, self.sample_points, -1)
        bs_cls_tokens = self.cls_token_mlp(bs_cls_tokens)
        # concat_emb.shape: (bs, num_points, 2*c)
        concat_emb = torch.concat([bs_cls_tokens, point_embed], dim=-1)
        # logits.shape: (bs, num_points)
        logits = self.finle_mlp(concat_emb)
        return logits, sample_points
    