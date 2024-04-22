# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Type

from models.sam.modeling import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        num_classes_outputs: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        encoder_embed_dim: int=1280
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.num_classes_outputs = num_classes_outputs

        # origin paramters
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # new paramters
        embed_dim, out_chans = encoder_embed_dim, transformer_dim
        self.process_inter_feat = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        self.upscaling_inter_feat = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )
        self.output_upscaling_2x = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            activation(),
        )
        self.cls_tokens = nn.Embedding(self.num_classes_outputs, transformer_dim)
        self.cls_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 16, 3)
                for i in range(self.num_classes_outputs)
            ]
        )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        inter_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        predict masks for <image_embeddings, prompt> pairs.

        The pairs will be automatically generated in the following cases:
            1. one-vs-all: In single image with multiple prompts, image embedding will be broadcasted to n_prompts
            2. all-vs-one: In multiple images with single prompt, prompt will be broadcasted to batch_size
        In multiple images with multiple prompts, the inputs will be regarded as pairs and not be broadcasted.

        Arguments:
            image_embeddings (torch.Tensor): [B1 x embed_dim x embed_h x embed_w] image embeddings.
            image_pe (torch.Tensor): [1 x embed_dim x embed_h x embed_w] position encoding. 
            According to the code, it is a learnable image-independent parameter.
            sparse_prompt_embeddings (torch.Tensor): [B2 x N x embed_dim] sparse prompt embeddings 
            with token length N. 
            dense_prompt_embeddings (torch.Tensor): [B2 x embed_dim x embed_h] dense prompt embeddings
        """
        # Concatenate output tokens
        # output_tokens = self.mask_tokens.weight
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.cls_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        b1, b2 = image_embeddings.shape[0], sparse_prompt_embeddings.shape[0]
        if b1 > 1 and b2 == 1:
            tokens = torch.repeat_interleave(tokens, b1, dim=0)
            # dense_prompt_embeddings = torch.repeat_interleave(dense_prompt_embeddings, b1, dim=0)
        elif b2 > 1 and b1 == 1:
            image_embeddings = torch.repeat_interleave(image_embeddings, b2, dim=0)
        elif b1 != b2:
            raise ValueError(f'The input embeddings pairs cannot be automatically generated! Get image_embeddings with shape {image_embeddings.shape} and sparse_embeddings with shape {sparse_prompt_embeddings.shape}')

        src = image_embeddings + dense_prompt_embeddings
        b, c, h, w = src.shape
        pos_src = torch.repeat_interleave(image_pe, b, dim=0)
        
        # Run the transformer
        # hs, src,tow_way_keys = self.transformer(src, pos_src, tokens)
        # hs, src, attn_token_to_image = self.transformer(src, pos_src, tokens)
        inter_feat_c256 = self.process_inter_feat(inter_feature.permute(0, 3, 1, 2))
        hs, src = self.transformer(src, pos_src, tokens, inter_feat_c256)
        mask_tokens_out = hs[:, 0 : (1 + self.num_mask_tokens), :]
        start_idx,end_idx = 1 + self.num_mask_tokens, 1 + self.num_mask_tokens + self.num_classes_outputs
        cls_mask_tokens_out = hs[:, start_idx:end_idx, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        upscaled_inter_embedding = self.upscaling_inter_feat(inter_feature.permute(0, 3, 1, 2))
        fused_embedding = upscaled_embedding + upscaled_inter_embedding
        upscaled_embedding_2x = self.output_upscaling_2x(fused_embedding)

        cls_hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes_outputs):
            cls_hyper_in_list.append(self.cls_mlps[i](cls_mask_tokens_out[:, i, :]))
        cls_hyper_in = torch.stack(cls_hyper_in_list, dim=1)
        
        b, c, h, w = upscaled_embedding_2x.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        cls_mask = (cls_hyper_in @ upscaled_embedding_2x.view(b, c, h * w)).view(b, -1, h, w)

        return cls_mask


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
