import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sam.build_sam import sam_model_registry
import numpy as np
import cv2
from models.sam.utils.amg import (
    MaskData, calculate_stability_score, 
    build_point_grid,batch_iterator,batched_mask_to_box
)
from torchvision.ops.boxes import batched_nms


class ChannelProjectorNet(nn.Module):
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
        super(ChannelProjectorNet, self).__init__()
        self.mpoints = mpoints
        self.box_nms_thresh = box_nms_thresh
        self.points_per_batch = points_per_batch
        self.n_per_side = n_per_side

        sam = sam_model_registry['vit_h'](checkpoint = sam_ckpt)
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder

        transformer_dim = 256
        self.output_cls_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        
        point_grids = build_point_grid(n_per_side)
        self.points_for_sam = point_grids*1024

        self.freeze_parameters()


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

    def forward(self, bs_image_embedding: torch.Tensor, mask_256: torch.Tensor):
        '''
        Args:
        - bs_image_embedding: tensor, shape is (1, 256, 64, 64)
        - mask_256: tensor, shape is (1, 256, 256)
        '''
        sam_proposal_data = self.proposal_region_from_sam(bs_image_embedding, mask_256)
        keep_token_out = sam_proposal_data['mask_token_out']
        keep_embed_256 = sam_proposal_data['embed_256']
        keep_masks_256 = sam_proposal_data['masks']
        keeped_seg_gt = sam_proposal_data['seg_matched_gt']
        
        hyper_in_cls = self.output_cls_mlps(keep_token_out)
        b,c,h,w = keep_embed_256.shape[0],32,256,256
        # keep_cls_logits.shape: (keep_by_nms, 1, 256, 256)
        keep_cls_logits = (hyper_in_cls @ keep_embed_256.view(b, c, h * w)).view(b, -1, h, w)
       
        outputs = {
            'keeped_seg_gt': keeped_seg_gt,
            'keep_cls_logits': keep_cls_logits,
            'keep_masks_256': keep_masks_256
        }

        return outputs
    
    def forward_batch_points(self, sam_proposal_data: MaskData):

        keep_token_out = sam_proposal_data['mask_token_out']
        keep_embed_256 = sam_proposal_data['embed_256']
        
        hyper_in_cls = self.output_cls_mlps(keep_token_out)
        b,c,h,w = keep_embed_256.shape[0],32,256,256
        # keep_cls_logits.shape: (keep_by_nms, 1, 256, 256)
        keep_cls_logits = (hyper_in_cls @ keep_embed_256.view(b, c, h * w)).view(b, -1, h, w)
       
        outputs = {
            'keep_cls_logits': keep_cls_logits,
        }

        return outputs

    def proposal_region_from_sam(self, bs_image_embedding, mask_256):
        mask_threshold = 0
        pred_iou_thresh = 0.88
        stability_score_thresh = 0.95
        stability_score_offset = 1.0

        device = bs_image_embedding.device
        with torch.no_grad():
            image_pe = self.prompt_encoder.get_dense_pe()
            data = MaskData()
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
                # mask_token_out_bs.shape: (points_per_batch, 1, 256)
                low_res_logits_bs, iou_pred_bs, embeddings_64_bs, embeddings_256_bs, mask_token_out_bs = self.mask_decoder(
                    image_embeddings = bs_image_embedding,
                    image_pe = image_pe,
                    sparse_prompt_embeddings = sparse_embeddings_every,
                    dense_prompt_embeddings = dense_embeddings_every,
                )
                batch_data = MaskData(
                    masks = low_res_logits_bs.flatten(0, 1),    # shape: (points_per_batch, 256, 256)
                    iou_preds = iou_pred_bs.flatten(0, 1),  # shape: (points_per_batch,)
                    embed_256 = embeddings_256_bs,  # shape: (points_per_batch, 32, 256, 256)
                    mask_token_out = mask_token_out_bs,  # shape: (points_per_batch, 1, 256)
                    points = torch.as_tensor(point_batch.repeat(low_res_logits_bs.shape[1], axis=0)), # shape: (points_per_batch, 2)
                )
                
                # # Filter by predicted IoU
                # keep_mask = batch_data["iou_preds"] > pred_iou_thresh
                # batch_data.filter(keep_mask)

                # # Calculate stability score
                # batch_data["stability_score"] = calculate_stability_score(
                #     batch_data["masks"], mask_threshold, stability_score_offset
                # )
                # keep_mask = batch_data["stability_score"] >= stability_score_thresh
                # batch_data.filter(keep_mask)

                # Threshold masks and calculate boxes
                batch_data["masks"] = batch_data["masks"] > mask_threshold
                # seg_matched_gt.shape: (points_per_batch_keeped, 1, 256, 256)
                batch_data["seg_matched_gt"] = batch_data["masks"] & mask_256
                batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])

                data.cat(batch_data)
                del batch_data

            # keep_by_nms = batched_nms(
            #     data["boxes"].float(),
            #     data["iou_preds"],
            #     torch.zeros_like(data["boxes"][:, 0]),  # categories
            #     iou_threshold=self.box_nms_thresh,
            # )
            # data.filter(keep_by_nms)

        return data
        

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