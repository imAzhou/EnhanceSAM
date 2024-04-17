# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type

from models.sam import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_classes: int = 3,
        num_queries: int = 100,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_classes (int): the number of classes
          num_queries (int): Number of query in Transformer decoder
          activation (nn.Module): the type of activation to use when
            upscaling masks
          
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.mask_tokens = nn.Embedding(num_queries, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.cls_embed = nn.Linear(transformer_dim, self.num_classes)
        self.mask_embed = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 5)

        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_queries)
        #     ]
        # )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        return self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
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
        output_tokens = self.mask_tokens.weight
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
        query_out, src = self.transformer(src, pos_src, tokens)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # upscaled_embedding = self.output_upscaling_4x(upscaled_embedding)

        # hyper_in_list: List[torch.Tensor] = []
        # for i in range(self.num_queries):
        #     hyper_in_list.append(self.output_hypernetworks_mlps[i](query_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)

        hyper_in = self.mask_embed(query_out)
        b, c, h, w = upscaled_embedding.shape
        masks_pred = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        cls_pred = self.cls_embed(query_out)

        return cls_pred, masks_pred


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
