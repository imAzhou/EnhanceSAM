import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sam.build_sam import sam_model_registry
from typing import List
from models.sam.utils.amg import MaskData, build_point_grid, batch_iterator
import numpy as np

class AutoSegNet(nn.Module):
    """
    Args:
    - n_per_side (int): The number of points to be sampled along one side of 
                        the image. The total number of points is n_per_side**2.
    - mpoints (int): The number of points to be sampled in connect region 
    """

    def __init__(self, n_per_side = 32, points_per_batch = 64,
                 num_classes = 1, use_embed = False, device = None,
                 sam_ckpt = 'checkpoints_sam/sam_vit_h_4b8939.pth'
            ):
        super(AutoSegNet, self).__init__()

        self.points_per_batch = points_per_batch
        self.n_per_side = n_per_side
        self.use_embed = use_embed
        self.num_classes = num_classes
        self.device = device
        sam = sam_model_registry['vit_h'](checkpoint = sam_ckpt)
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.preprocess = sam.preprocess
        self.mask_decoder = sam.mask_decoder

        transformer_dim = sam.mask_decoder.transformer_dim
        self.output_cls_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_classes)
            ]
        )
        
        point_grids = build_point_grid(n_per_side)
        self.points_for_sam = point_grids*1024

        self.freeze_parameters()

    def freeze_parameters(self):
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        # freeze mask encoder
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        
    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        torch.save(self.output_cls_mlps.state_dict(), filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)
        print(self.output_cls_mlps.load_state_dict(state_dict))

    def gene_img_embed(self, sampled_batch: dict):
        if self.use_embed:
            bs_image_embedding = sampled_batch['img_embed'].to(self.device)
        else:
            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                input_images = self.preprocess(sampled_batch['input_image']).to(self.device)
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                bs_image_embedding = self.image_encoder(input_images, need_inter=False).detach()
        return bs_image_embedding
    
    def forward_batch_points(self, bs_image_embedding: torch.Tensor, point_batch: list):
        with torch.no_grad():
            bs_coords_torch_every = torch.as_tensor(np.array(point_batch), dtype=torch.float, device=self.device).unsqueeze(1)
            bs_labels_torch_every = torch.ones(bs_coords_torch_every.shape[:2], dtype=torch.int, device=self.device)
            points_every = (bs_coords_torch_every, bs_labels_torch_every)
            sparse_embeddings_every, dense_embeddings_every = self.prompt_encoder(
                points=points_every,
                boxes=None,
                masks=None,
            )
            # low_res_logits_every.shape: (points_per_batch, 1, 256, 256)
            # iou_pred_every.shape: (points_per_batch, 1)
            # embeddings_256_every.shape: (points_per_batch, 32, 256, 256)
            # mask_token_out_bs.shape: (points_per_batch, num_class, 256)
            low_res_logits_bs, iou_pred_bs, embeddings_64_bs, embeddings_256_bs, mask_token_out_bs = self.mask_decoder(
                image_embeddings = bs_image_embedding,
                image_pe = self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse_embeddings_every,
                dense_prompt_embeddings = dense_embeddings_every,
            )
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(self.output_cls_mlps[i](mask_token_out_bs[:, i, :]))
        hyper_in_cls = torch.stack(hyper_in_list, dim=1)
        b,c,h,w = embeddings_256_bs.shape[0],32,256,256
        # cls_logits.shape: (points_per_batch, 1, 256, 256)
        cls_logits = (hyper_in_cls @ embeddings_256_bs.view(b, c, h * w)).view(b, -1, h, w)

        outputs = {
            'cls_logits': cls_logits,
            'sam_logits': low_res_logits_bs,
            'batch_points': bs_coords_torch_every.squeeze(1).cpu()
        }

        return outputs


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