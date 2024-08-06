import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy, calc_loss,show_mask,show_multi_mask
from models.two_stage_seg import TwoStageNet
from datasets.panoptic.create_loader import gene_loader_trainval
from mmengine.config import Config
from utils.metrics import get_metrics
from tqdm import tqdm
import torch.nn.functional as F
from mmengine.structures import PixelData
import numpy as np
from mmdet.evaluation.functional import INSTANCE_OFFSET
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import linear_sum_assignment

from utils.ssim_loss import SSIM
from utils.cl_dice import soft_dice_cldice
from utils.losses import BoundaryDoULoss
from prettytable import PrettyTable

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save_each_epoch', action='store_true')

args = parser.parse_args()

def onehot2instmask(onehot_mask, with_boundary=False):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    inst_boundary_mask = np.zeros((len(onehot_mask), h, w))
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
        if with_boundary:
            dilated_mask = binary_dilation(seg)
            eroded_mask = binary_erosion(seg)
            boundary_mask = dilated_mask ^ eroded_mask   # binary mask (h,w)
            inst_boundary_mask[iid] = boundary_mask
    
    if with_boundary:
        inst_overlap_boundary_mask = np.sum(inst_boundary_mask, axis=0)
        inst_overlap_boundary_mask = (inst_overlap_boundary_mask > 1).astype(int)
        inst_overlap_boundary_mask = binary_dilation(inst_overlap_boundary_mask, iterations=2)
        inst_boundary_mask = np.max(inst_boundary_mask, axis=0)
        inst_boundary_mask[inst_overlap_boundary_mask] = 1
        return instmask, inst_boundary_mask
    
    return instmask

def get_fast_pq(true, pred, match_iou=0.5):
        """`match_iou` is the IoU threshold level to determine the pairing between
        GT instances `p` and prediction instances `g`. `p` and `g` is a pair
        if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
        (1 prediction instance to 1 GT instance mapping).

        If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
        in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

        If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
        the number of pairs is also maximal.    
        
        Fast computation requires instance IDs are in contiguous orderding 
        i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
        and `by_size` flag has no effect on the result.

        Returns:
            [dq, sq, pq]: measurement statistic

            [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                        pairing information to perform measurement
                        
        """
        assert match_iou >= 0.0, "Cant' be negative"

        true = np.copy(true)
        pred = np.copy(pred)
        true_id_list = list(np.unique(true))
        pred_id_list = list(np.unique(pred))

        true_masks = [
            None,
        ]
        for t in true_id_list[1:]:
            t_mask = np.array(true == t, np.uint8)
            true_masks.append(t_mask)

        pred_masks = [
            None,
        ]
        for p in pred_id_list[1:]:
            p_mask = np.array(pred == p, np.uint8)
            pred_masks.append(p_mask)

        # prefill with value
        pairwise_iou = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        # caching pairwise iou
        for true_id in true_id_list[1:]:  # 0-th is background
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id - 1, pred_id - 1] = iou
        #
        if match_iou >= 0.5:
            paired_iou = pairwise_iou[pairwise_iou > match_iou]
            pairwise_iou[pairwise_iou <= match_iou] = 0.0
            paired_true, paired_pred = np.nonzero(pairwise_iou)
            paired_iou = pairwise_iou[paired_true, paired_pred]
            paired_true += 1  # index is instance id - 1
            paired_pred += 1  # hence return back to original
        else:  # * Exhaustive maximal unique pairing
            #### Munkres pairing with scipy library
            # the algorithm return (row indices, matched column indices)
            # if there is multiple same cost in a row, index of first occurence
            # is return, thus the unique pairing is ensure
            # inverse pair to get high IoU as minimum
            paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
            ### extract the paired cost and remove invalid pair
            paired_iou = pairwise_iou[paired_true, paired_pred]

            # now select those above threshold level
            # paired with iou = 0.0 i.e no intersection => FP or FN
            paired_true = list(paired_true[paired_iou > match_iou] + 1)
            paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
            paired_iou = paired_iou[paired_iou > match_iou]

        # get the actual FP and FN
        unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
        unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

        #
        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
        # get the SQ, no paired has 0 iou so not impact
        sq = paired_iou.sum() / (tp + 1.0e-6)

        return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

def gene_insts_color(data_sample, pan_result, boundary):
    img_path = data_sample.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = data_sample.gt_sem_seg.sem_seg

    h,w = 256,256
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(pan_result)) - 1
    
    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(131)
    ax.imshow(img)
    show_mask(gt.cpu(), ax, rgb=[47, 243, 15])
    ax.set_axis_off()
    ax.set_title('gt mask')

    ax = fig.add_subplot(132)
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[pan_result==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_axis_off()
    ax.set_title('color gt')
    
    ax = fig.add_subplot(133)
    ax.imshow(boundary, cmap='gray')
    ax.set_axis_off()
    ax.set_title('gt boundary')

    plt.tight_layout()
    dirname = 'visual_results/color_boundary'
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f'{dirname}/{image_name}.png')
    plt.close()

def draw_pred(
        datainfo, gt_mask, pred_mask, gt_inst_mask, pred_inst_mask, gt_boundary, pred_boundary):
    '''
    Args:
        pred_mask: tensor, shape is (h,w), h=w=1024
        points: tensor, (num_points, 2), 2 is [x,y]
        boxes: tensor, , (num_boxes, 4), 4 is [x1,y1,x2,y2]
    '''
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w = 256,256

    rgbs = [[255,255,255],[47, 243, 15]]

    fig = plt.figure(figsize=(12,9))
    
    ax = fig.add_subplot(231)
    ax.imshow(img)
    show_multi_mask(gt_mask.cpu(), ax, palette = rgbs)
    ax.set_title('gt mask')
    ax = fig.add_subplot(232)
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(gt_inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[gt_inst_mask==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_axis_off()
    ax.set_title('color gt')
    ax = fig.add_subplot(233)
    ax.imshow(gt_boundary, cmap='gray')
    ax.set_title('gt boundary')
    
    ax = fig.add_subplot(234)
    ax.imshow(img)
    show_multi_mask(pred_mask.cpu(), ax, palette = rgbs)
    ax.set_title('pred mask')
    ax = fig.add_subplot(235)
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(pred_inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[pred_inst_mask==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_title('color pred')
    ax = fig.add_subplot(236)
    ax.imshow(pred_boundary, cmap='gray')
    ax.set_title('pred boundary')

    plt.tight_layout()
    plt.savefig(f'{image_name}')
    plt.close()

def train_one_epoch(model: TwoStageNet, train_loader, optimizer, logger, epoch_num, cfg):

    model.train()
    len_loader = len(train_loader)
    ssim_loss_fn = SSIM(window_size=11,size_average=True)
    cldice_loss_fn = soft_dice_cldice(iter_=50)
    bIoU_loss_fn = BoundaryDoULoss()

    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        gt_cell_masks = []
        gt_boundary_masks = []
        for data_sample in sampled_batch['data_samples']:
            gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            # tensor, (h, w)
            gt_inst_mask,gt_boundary_mask = onehot2instmask(gt_inst_seg, with_boundary=True)
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg.squeeze(0)    # tensor, (h, w) pixel value is cls id
            gt_boundary_mask = torch.as_tensor(gt_boundary_mask, dtype=gt_sem_seg.dtype)
            gt_cell_masks.append(gt_sem_seg.unsqueeze(0))
            gt_boundary_masks.append(gt_boundary_mask.unsqueeze(0))

        gt_cell_masks = torch.cat(gt_cell_masks, dim = 0).to(device).float()    # (bs, h, w)
        gt_boundary_masks = torch.cat(gt_boundary_masks, dim = 0).to(device).float()    # (bs, h, w)
        
        outputs = model.forward_coarse(sampled_batch)
        logits = outputs['logits_origin']  # (bs, k, o_h, o_w)

        bs,c,h,w = logits.shape
        pred_logits = logits.reshape(-1, h, w).unsqueeze(1)
        targets = torch.stack((gt_cell_masks, gt_boundary_masks),dim=1).reshape(-1, h, w)
        bce_dice_loss = calc_loss(pred_logits=pred_logits, target_masks=targets, args=cfg)
        # ssim_loss = 1 - ssim_loss_fn(F.sigmoid(pred_logits),targets.unsqueeze(1))
        # boundary_logits = F.sigmoid(logits[:,1,:,:].unsqueeze(1))
        # boundary_target = gt_boundary_masks.unsqueeze(1)
        # bIoU_loss = bIoU_loss_fn(boundary_logits, boundary_target)
        # loss = bce_dice_loss + bIoU_loss
        loss = bce_dice_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i_batch+1) % args.print_interval == 0:
            # , bIoU_loss: {bIoU_loss.item():.6f}
            info_str = f'iteration {i_batch+1}/{len_loader}, bce_dice_loss: {bce_dice_loss.item():.6f}'
            logger.info(info_str)

def val_one_epoch(model: TwoStageNet, val_loader, evaluators, epoch_num, cfg):
    
    model.eval()
    all_PQs = []    #[[dq, sq, pq],...,[dq, sq, pq]]
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        outputs = model.forward_coarse(sampled_batch)
        logits = outputs['logits_origin']  # (bs or k_prompt, 3, 256, 256)

        for idx,datainfo in enumerate(sampled_batch['data_samples']):
            gt_inst_seg = datainfo.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask,gt_boundary_mask = onehot2instmask(gt_inst_seg,with_boundary=True)    # tensor, (h, w)

            pred_cell_mask = (logits[idx,0,:,:].detach() > 0).type(torch.uint8)
            pred_boundary_mask = logits[idx,1,:,:].detach() > 0

            pred_cell_mask[pred_boundary_mask] = 0
            pred_inst_mask = evaluators['inst_evaluator'].find_connect(pred_cell_mask, dilation=True)
            PQs,_ = get_fast_pq(gt_inst_mask, pred_inst_mask)
            all_PQs.append(PQs)
    mean_PQs = np.array(all_PQs).mean(axis=0)
    return mean_PQs


def main(logger_name, cfg):
    # load datasets
    train_dataloader, val_dataloader, metainfo, restinfo = gene_loader_trainval(
        dataset_config = d_cfg, seed = args.seed)

    # register model
    model = TwoStageNet(
                num_classes = cfg.num_classes,
                num_mask_tokens = cfg.num_mask_tokens,
                sm_depth = cfg.semantic_module_depth,
                use_inner_feat = cfg.use_inner_feat,
                use_boundary_head = cfg.use_boundary_head,
                use_embed = cfg.dataset.load_embed,
                sam_ckpt = cfg.sam_ckpt,
                sam_type = cfg.sam_type,
                device = device
            ).to(device)
    
    # create logger
    logger,files_save_dir = get_logger(
        args.record_save_dir, model, cfg, logger_name)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    # get optimizer and lr_scheduler
    optimizer,lr_scheduler = get_train_strategy(model, cfg)
    # get evaluator
    evaluators = get_metrics(['pixel', 'inst'], metainfo, restinfo)

    # train and val in each epoch
    all_metrics,all_value = [],[]
    max_value,max_epoch = 0,0
    for epoch_num in range(cfg.max_epochs):
        # train
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"epoch: {epoch_num}, learning rate: {current_lr:.6f}")
        train_one_epoch(model, train_dataloader, optimizer, logger, epoch_num, cfg)
        lr_scheduler.step()
        if cfg.save_each_epoch:
            save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
            model.save_parameters(save_mode_path)
            logger.info(f"save model to {save_mode_path}")

        # val
        table_data = PrettyTable()
        # PQs: [dq, sq, pq]
        PQs = val_one_epoch(model, val_dataloader, evaluators, epoch_num, cfg)
        metrics = dict(
            PQ = PQs[2], SQ = PQs[1], DQ = PQs[0],
        )
        for key,value in metrics.items():
            table_data.add_column(key,[f'{value:.4f}'])
        logger.info('\n' + table_data.get_string())
        
        best_score = PQs[2] # PQ
        # logger.info(f'epoch: {epoch_num} ' + str(metrics) + '\n')
        if best_score > max_value:
            max_value = best_score
            max_epoch = epoch_num
            save_mode_path = os.path.join(pth_save_dir, 'best.pth')
            model.save_parameters(save_mode_path)

        all_value.append(best_score)
        all_metrics.append(f'epoch: {epoch_num}' + str(metrics) + '\n')
    
    print(f'max_value: {max_value}, max_epoch: {max_epoch}')
    # save result file
    config_file = os.path.join(files_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_metrics)
        f.write(f'\nmax_value: {max_value}, max_epoch: {max_epoch}\n')
        f.write(str(all_value))


if __name__ == "__main__":
    device = torch.device(args.device)

    d_cfg = Config.fromfile(args.dataset_config_file)
    model_strategy_config_file = 'configs/model_strategy.py'
    m_s_cfg = Config.fromfile(model_strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, m_s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    search_lr = cfg.get('search_lr', None)
    search_loss_type = cfg.get('search_loss_type', None)
    for loss_type in search_loss_type:
        cfg.loss_type = loss_type
        for idx,base_lr in enumerate(search_lr):
            cfg.base_lr = base_lr
            set_seed(args.seed)
            logger_name = f'lr_{loss_type}_{idx}'
            main(logger_name, cfg)



'''
python scripts/panoptic/binary/coarse_main_3.py \
    configs/datasets/pannuke_binary.py \
    --record_save_dir logs/0_pannuke_binary_02 \
    --print_interval 20
    --save_each_epoch
'''
