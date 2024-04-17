import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.losses import DiceLoss
from help_func.tools import one_hot_encoder


class TaskHead(nn.Module):
    """
    Args:
        sam_model: a vision transformer model, see base_vit.py
        num_classes: how many classes the model output, default to the vit model
    """

    def __init__(self, 
                num_classes: int = 3,
                num_queries: int = 100,
                ):
        super(TaskHead, self).__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.ce_loss_fn = CrossEntropyLoss(ignore_index = 255)
        self.dice_loss_fn = DiceLoss(num_classes, ignore_index = 255)

    def forward(self, 
                cls_pred: Tensor, 
                masks_pred: Tensor,
                gt_label_batch: Tensor,
            ):
        '''
        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
        '''
        # upsample mask
        mask_pred_results = F.interpolate(
            masks_pred, 
            size=1024, 
            mode='bilinear', 
            align_corners=False)
        cls_score = F.softmax(cls_pred, dim=-1)
        mask_pred = mask_pred_results.sigmoid()
        # seg_logits.shape: (bs, num_class, h, w)
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        # target_one_hot = one_hot_encoder(self.num_classes, gt_label_batch)
        
        ce_loss = self.ce_loss_fn(seg_logits, gt_label_batch.long())
        dice_loss = self.dice_loss_fn(seg_logits,gt_label_batch, softmax=True)

        return dict(
            ce_loss = 0.3*ce_loss,
            dice_loss = 0.7*dice_loss,
        )