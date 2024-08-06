import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from mmdet.models.losses import FocalLoss,DiceLoss
from .loss_mask import loss_masks


def calc_loss(*, pred_logits, target_masks, args):
    '''
    Args:
        pred_logits: (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for class-specific predict logits.
        target_masks: A tensor of shape (N, H, W) that contains class index on a H x W grid, 0 mean bg, valued class id from 1 to 2,3,4....,C-1
    '''

    if args.loss_type == 'loss_masks':  # sample points calc loss, only for binary mask
        bce_loss, dice_loss = loss_masks(pred_logits, target_masks.unsqueeze(1))
        loss = (1 - args.dice_param) * bce_loss + args.dice_param * dice_loss
        return loss
    if args.loss_type == 'focal_dice':  # only for binary mask
        return focal_dice(pred_logits, target_masks.unsqueeze(1), args)
    if args.loss_type == 'ce_dice': # calc every pixels loss
        return ce_dice(pred_logits, target_masks, args)


def focal_dice(pred_logits, target_masks, args):
    focal_loss_fn = FocalLoss()
    bdice_loss_fn = DiceLoss()
    bce_loss = focal_loss_fn(pred_logits, target_masks)
    dice_loss = bdice_loss_fn(pred_logits.flatten(1), target_masks.flatten(1))
    loss = (1 - args.dice_param) * bce_loss + args.dice_param * dice_loss
    return loss

def ce_dice(pred_logits, target_masks, args):
    '''
    for CrossEntropyLoss, target_masks each pixel record its class id,
    for DiceLoss, target_masks shoud be converted to onehot format.
    '''
    ignore_index = 255
    ce_loss_fn = CrossEntropyLoss(ignore_index = ignore_index)
    ce_loss = ce_loss_fn(pred_logits, target_masks.long())

    dice_loss_fn = DiceLoss()
    cls_nums = pred_logits.shape[1]
    gt_c_mask = torch.zeros_like(pred_logits).to(pred_logits.device)
    for cid in range(cls_nums):
        gt_c_mask[:,cid,:,:] = (target_masks == cid).int()
    dice_loss = dice_loss_fn(pred_logits,gt_c_mask)
    if torch.sum(target_masks != 255) != 0:
        loss = (1 - args.dice_param) * ce_loss + args.dice_param * dice_loss
    else:
        loss = torch.tensor(0.0, requires_grad=True)
    return loss

def calc_cls_loss_fn(pred_logits, target_labels):
    '''
    Args:
        - pred_logits: shape is (bs,k)
        - target_labels: shape is (bs,k)
    '''
    loss_fn = BCEWithLogitsLoss()
    cls_loss = loss_fn(pred_logits.flatten(0,1), target_labels[:].float())
    return 


class BoundaryDoULoss(nn.Module):
    def __init__(self):
        super(BoundaryDoULoss, self).__init__()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        '''
        Args:
            - inputs: logits after sigmoid, shape is (bs, 1, h, w)
            - target: tensor, float, shape is (bs, 1, h, w)
        '''

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = self._adaptive_size(inputs[:, 0], target[:, 0])
        
        return loss
  