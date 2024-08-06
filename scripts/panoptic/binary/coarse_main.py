import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy, calc_loss
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

from utils.ssim_loss import SSIM

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--metrics', nargs='*', type=str, default=['pixel', 'inst'])
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

def onehot2instmask(onehot_mask):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
    
    return instmask

# 提取边界函数
def extract_boundary(mask):
    dilated_mask = binary_dilation(mask)
    eroded_mask = binary_erosion(mask)
    boundary = dilated_mask ^ eroded_mask
    return boundary


def train_one_epoch(model: TwoStageNet, train_loader, optimizer, logger, cfg):

    model.train()
    len_loader = len(train_loader)
    # ssim_loss_fn = SSIM(window_size=11,size_average=True)

    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        gt_mask = []
        origin_h,origin_w = sampled_batch['data_samples'][0].ori_shape
        for data_sample in sampled_batch['data_samples']:
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            gt_mask.append(gt_sem_seg)
        gt_mask = torch.cat(gt_mask, dim = 0).to(device)    # (bs, h, w)
        origin_size = sampled_batch['data_samples'][0].ori_shape
        gt_mask = F.interpolate(gt_mask.unsqueeze(1).float(), origin_size).squeeze(1)
        outputs = model.forward_coarse(sampled_batch)
        logits = outputs['logits_origin']  # (bs or k_prompt, 1, h, w)
        
        target_masks = gt_mask.reshape(-1, origin_h, origin_w)
        pred_logits = logits.reshape(-1, origin_h, origin_w).unsqueeze(1)
        bce_dice_loss = calc_loss(pred_logits=pred_logits, target_masks=target_masks, args=cfg)
        # ssim_loss = 1 - ssim_loss_fn(F.sigmoid(pred_logits),target_masks.unsqueeze(1))
        loss = bce_dice_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i_batch+1) % args.print_interval == 0:
            logger.info(f'iteration {i_batch+1}/{len_loader}, bce_dice_loss: {bce_dice_loss.item():.6f}')

def val_one_epoch(model: TwoStageNet, val_loader, evaluators, cfg):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):

        origin_h,origin_w = sampled_batch['data_samples'][0].ori_shape
        outputs = model.forward_coarse(sampled_batch)
        logits = outputs['logits_origin']  # (bs or k_prompt, 1, origin_h, origin_w)
        bs_pred_origin = (logits>0).detach().type(torch.uint8)

        all_datainfo = []
        for idx,datainfo in enumerate(sampled_batch['data_samples']):
            datainfo.pred_sem_seg = PixelData(sem_seg=bs_pred_origin[idx])
            
            if 'inst' in args.metrics or 'panoptic' in args.metrics:
                gt_inst_seg = datainfo.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
                gt_inst_mask = onehot2instmask(gt_inst_seg)    # tensor, (h, w)
                pred_cell_mask = (logits[idx,0,:,:].detach() > 0).type(torch.uint8)
                pred_inst_mask = evaluators['inst_evaluator'].find_connect(pred_cell_mask)
                if 'inst' in args.metrics:
                    evaluators['inst_evaluator'].process(gt_inst_mask, pred_inst_mask)
                panoptic_seg = torch.full((origin_h,origin_w), 0, dtype=torch.int32, device=device)
                if np.sum(pred_inst_mask) > 0:
                    iid = np.unique(pred_inst_mask)
                    for i in iid[1:]:
                        mask = pred_inst_mask == i
                        panoptic_seg[mask] = (1 + i * INSTANCE_OFFSET)
                datainfo.pred_panoptic_seg = PixelData(sem_seg=panoptic_seg[None])

            all_datainfo.append(datainfo.to_dict())
        
        evaluators['semantic_evaluator'].process(None, all_datainfo)
        if 'panoptic' in args.metrics:
            evaluators['panoptic_evaluator'].process(None, all_datainfo)
    
    total_datas = len(val_loader)*cfg.val_bs
    semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
    metrics = dict(
        semantic = semantic_metrics,
    )
    
    if 'inst' in args.metrics:
        inst_metrics = evaluators['inst_evaluator'].evaluate(total_datas)
        metrics['inst'] = inst_metrics
    if 'panoptic' in args.metrics:
        panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
        metrics['panoptic'] = panoptic_metrics
    return metrics


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
    evaluators = get_metrics(args.metrics, metainfo, restinfo)

    # train and val in each epoch
    all_metrics,all_value = [],[]
    max_value,max_epoch = 0,0
    for epoch_num in range(cfg.max_epochs):
        # train
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"epoch: {epoch_num}, learning rate: {current_lr:.6f}")
        train_one_epoch(model, train_dataloader, optimizer, logger, cfg)
        lr_scheduler.step()
        if cfg.save_each_epoch:
            save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
            model.save_parameters(save_mode_path)
            logger.info(f"save model to {save_mode_path}")

        # val
        metrics = val_one_epoch(model, val_dataloader, evaluators, cfg)
        # if 'inst' in args.metrics:
        #     best_score = metrics['inst']['AJI']
        # if 'panoptic' in args.metrics:
        #     best_score = metrics['panoptic']['coco_panoptic/PQ_th']
        # else:
        best_score = metrics['semantic']['mIoU']
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
# 02: 不更新 output_upscaling, 没有 ssim loss, mobilev2 结构没有残差连接
python scripts/panoptic/binary/coarse_main.py \
    configs/datasets/mass.py \
    --record_save_dir logs/ablation/mass \
    --metrics pixel
'''
