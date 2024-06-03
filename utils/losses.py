import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from .loss_mask import loss_masks

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=2, weight=None, size_average=None,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.balance_param = balance_param

    def forward(self, input, target):
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=self.weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        return self.balance_param * focal_loss

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, C, H, W]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(*predict.shape[:2], -1)
        predict = predict.sigmoid()
        target = target.contiguous().view(*target.shape[:2], -1)

        num = torch.sum(torch.mul(predict, target), dim=-1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=-1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=None, ):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            if i != self.ignore_index:
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


def cls_loss(*, pred_logits, target_masks, args):
    '''
    Args:
        pred_logits: (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for class-specific predict logits.
        target_masks: A tensor of shape (N, H, W) that contains class index on a H x W grid.
    '''
    if args.loss_type == 'loss_masks':
        bce_loss, dice_loss =  loss_masks(pred_logits, target_masks.unsqueeze(1).float())
        return bce_loss + dice_loss
    if args.loss_type == 'bce_bdice':
        return bce_bdice(pred_logits, target_masks, args)
    if args.loss_type == 'focal_bdice':
        return focal_bdice(pred_logits, target_masks, args)
    if args.loss_type == 'ce_dice':
        return ce_dice(pred_logits, target_masks, args)

def bce_bdice(pred_logits, target_masks, args):
    bce_loss_fn = BCEWithLogitsLoss()
    bdice_loss_fn = BinaryDiceLoss()
    bce_loss = bce_loss_fn(pred_logits.squeeze(1), target_masks[:].float())
    dice_loss = bdice_loss_fn(pred_logits, target_masks.unsqueeze(1))
    loss = (1 - args.dice_param) * bce_loss + args.dice_param * dice_loss
    return loss

def focal_bdice(pred_logits, target_masks, args):
    focal_loss_fn = FocalLoss()
    bdice_loss_fn = BinaryDiceLoss()
    bce_loss = focal_loss_fn(pred_logits.squeeze(1), target_masks[:].float())
    dice_loss = bdice_loss_fn(pred_logits, target_masks.unsqueeze(1))
    loss = (1 - args.dice_param) * bce_loss + args.dice_param * dice_loss
    return loss

def ce_dice(pred_logits, target_masks, args):
    ce_loss_fn = CrossEntropyLoss(ignore_index = 255)
    dice_loss_fn = DiceLoss(args.num_classes, ignore_index = 255)
    ce_loss = ce_loss_fn(pred_logits, target_masks.long())
    dice_loss = dice_loss_fn(pred_logits,target_masks, softmax=True)
    if torch.sum(target_masks != 255) != 0:
        loss = (1 - args.dice_param) * ce_loss + args.dice_param * dice_loss
    else:
        loss = torch.tensor(0.0, requires_grad=True)
    return loss