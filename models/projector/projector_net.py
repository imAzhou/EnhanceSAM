import torch
import torch.nn as nn
import torch.nn.functional as F
from help_func.build_sam import sam_model_registry
import numpy as np
import cv2
from models.sam.utils.amg import build_point_grid,batch_iterator
from models.sam import Attention, MLPBlock


class ProjectorNet(nn.Module):
    """
    Args:
    - n_per_side (int): The number of points to be sampled along one side of 
                        the image. The total number of points is n_per_side**2.
    - mpoints (int): The number of points to be sampled in connect region 
    """

    def __init__(self, n_per_side = 32, points_per_batch = 64,
                 mpoints = 2,
                 box_nms_thresh: float = 0.7,
                 sam_ckpt = 'checkpoints_sam/sam_vit_h_4b8939.pth'):
        super(ProjectorNet, self).__init__()
        self.mpoints = mpoints
        self.box_nms_thresh = box_nms_thresh
        self.points_per_batch = points_per_batch
        self.n_per_side = n_per_side

        sam = sam_model_registry['vit_h'](checkpoint = sam_ckpt)
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        in_channels, out_channels = 33, 32
        vector_dim = 32
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=3, groups=in_channels, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )
        self.front_embed = nn.Embedding(1, vector_dim)
        self.cross_attn_front_to_kconnect = Attention(vector_dim, 1)
        self.norm1 = nn.LayerNorm(vector_dim)
        self.mlp = MLPBlock(vector_dim, vector_dim*4, nn.ReLU)
        self.norm2 = nn.LayerNorm(vector_dim)

        point_grids = build_point_grid(n_per_side)
        self.points_for_sam = point_grids*1024

        self.freeze_parameters()

    def get_kconnect_mpoints(self, binary_mask: np.ndarray, device):
        scaler = 1024 // 64
        # 定义侵蚀的结构元素
        kernel = np.ones((2, 2), np.uint8)
        binary_mask *= 255
        eroded_image = cv2.erode(binary_mask, kernel, iterations=1)
        # 获取图中所有连通区域，并在每个连通区域中取 mpoints 个前景点
        # step1: 获取所有连通区域
        num_labels, labels = cv2.connectedComponents(eroded_image)

        # 训练模式下从所有的连通区域内取 mpoints 个前景点
        # 推理模式下只取其中一个连通区域的 mpoints 个前景点
        chosen_k = num_labels-1 if self.training else 1
        # chosen_k = num_labels-1
        # step2: 随机选择k个连通区域，从它的像素坐标中随机选取 mpoints 个前景点
        choosen_k_idx = np.random.choice(range(1, num_labels), chosen_k, replace=False)

        bs_points = []
        for label in choosen_k_idx:
            region = np.argwhere(labels == label).tolist()
            # 随机获取mpoints个前景点坐标, 坐标格式：(y,x)
            random_positive_idx = np.random.choice(range(len(region)), self.mpoints, replace=True)
            # 将坐标放大到 1024 * 1024 尺度后送入 prompt encoder, 坐标格式：(x,y)
            coordinate_1024 = [[region[idx][1]*scaler, region[idx][0]*scaler] for idx in random_positive_idx]
            coords_torch = torch.as_tensor(coordinate_1024, dtype=torch.float, device=device)
            bs_points.append(coords_torch[None, :, :])
        bs_coords_torch = torch.cat(bs_points, dim = 0)
        bs_labels_torch = torch.ones(bs_coords_torch.shape[:2], dtype=torch.int, device=device)
        return bs_coords_torch, bs_labels_torch

    def freeze_parameters(self):
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        # freeze mask encoder
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        
    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        # merged_dict = {
        #     'projector': self.projector.state_dict()
        # }
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)
        # self.projector.load_state_dict(state_dict['projector'])
        print(self.load_state_dict(state_dict))

    def forward(self, bs_image_embedding: torch.Tensor, mask_64: torch.Tensor):
        device = bs_image_embedding.device
        points = self.get_kconnect_mpoints(mask_64[0].cpu().numpy(), device)
        image_pe = self.prompt_encoder.get_dense_pe()
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        # low_res_logits.shape: (kconnect, 1, 256, 256)
        low_res_logits, iou_pred, embeddings_64, embeddings_256 = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings = dense_embeddings,
        )
        kconnect_embedding = torch.cat([embeddings_256, low_res_logits>0], dim=1)
        kconnect_projection = self.projector(kconnect_embedding)
        kconnect_vector = torch.mean(kconnect_projection, dim=(2,3)).unsqueeze(0)
        front_vector = torch.mean(kconnect_vector, dim=1, keepdim=True)
        
        # q = self.front_embed.weight.unsqueeze(0)
        # attn_out = self.cross_attn_front_to_kconnect(q=q, k=kconnect_vector, v=kconnect_vector)
        # front_vector = q + attn_out
        # front_vector = self.norm1(front_vector)
        # mlp_out = self.mlp(front_vector)
        # front_vector = front_vector + mlp_out
        # front_vector = self.norm2(front_vector)

        every_projection = []
        for (point_batch,) in batch_iterator(self.points_per_batch, self.points_for_sam):
            bs_coords_torch_every = torch.as_tensor(point_batch, dtype=torch.float, device=device).unsqueeze(1)
            bs_labels_torch_every = torch.ones(bs_coords_torch_every.shape[:2], dtype=torch.int, device=device)
            points_every = (bs_coords_torch_every, bs_labels_torch_every)
            sparse_embeddings_every, dense_embeddings_every = self.prompt_encoder(
                points=points_every,
                boxes=None,
                masks=None,
            )
            # low_res_logits_every.shape: (points_per_batch, 1, 256, 256)
            # iou_pred_every.shape: (points_per_batch, 1)
            # embeddings_256_every.shape: (points_per_batch, 32, 256, 256)
            low_res_logits_bs, iou_pred_bs, embeddings_64_bs, embeddings_256_bs = self.mask_decoder(
                image_embeddings = bs_image_embedding,
                image_pe = image_pe,
                sparse_prompt_embeddings = sparse_embeddings_every,
                dense_prompt_embeddings = dense_embeddings_every,
            )


            every_embedding = torch.cat([embeddings_256_bs, low_res_logits_bs>0], dim=1)
            every_projection_batch = self.projector(every_embedding)
            every_projection.append(every_projection_batch)
        
        every_projection = torch.cat(every_projection, dim=0)
        every_vector = torch.mean(every_projection, dim=(2,3))

        front_vector = torch.repeat_interleave(front_vector.squeeze(0), every_vector.shape[0], dim=0)
        # cosine_similarity.shape: (n_per_side**2, )
        cosine_similarity = torch.cosine_similarity(front_vector, every_vector, dim=1)
        
        simi_logits = F.interpolate(
            cosine_similarity.view(self.n_per_side,self.n_per_side).unsqueeze(0).unsqueeze(0),
            (64, 64),
            mode="bilinear",
            align_corners=False,
        )

        outputs = {
            'simi_logits': simi_logits*5,
            'max_cos_simi': torch.max(cosine_similarity).item(),
            'min_cos_simi': torch.min(cosine_similarity).item()
        }

        return outputs
    