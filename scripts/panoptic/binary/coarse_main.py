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

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
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

# 生成边界掩码函数
def get_boundary_mask(mask):
    boundary = extract_boundary(mask)
    return torch.from_numpy(boundary).float()

# 可视化函数
def visualize_boundary(original, boundary):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original)
    
    plt.subplot(1, 2, 2)
    plt.title('Boundary Image')
    plt.imshow(boundary, cmap='gray')
    
    plt.savefig('boundary.png')

def train_one_epoch(model: TwoStageNet, train_loader, optimizer, logger, cfg):

    model.train()
    len_loader = len(train_loader)
    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        gt_mask = []
        for data_sample in sampled_batch['data_samples']:
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            gt_mask.append(gt_sem_seg)
        gt_mask = torch.cat(gt_mask, dim = 0).to(device)    # (bs, h, w)
        if gt_mask.shape[-1] != 256:
            gt_mask = F.interpolate(gt_mask.float().unsqueeze(1), (256, 256), mode="nearest").squeeze(1)
        outputs = model.forward_coarse(sampled_batch, cfg.train_prompt_type)
        logits = outputs['logits_256']  # (bs or k_prompt, 1, 256, 256)
        
        bs1,bs2 = gt_mask.shape[0], logits.shape[0]
        if bs2 > bs1:
            gt_mask = torch.repeat_interleave(gt_mask, bs2, dim=0)
        loss = calc_loss(pred_logits=logits, target_masks=gt_mask, args=cfg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i_batch+1) % args.print_interval == 0:
            logger.info(f'iteration {i_batch+1}/{len_loader}: loss: {loss.item():.6f}')

def val_one_epoch(model: TwoStageNet, val_loader, evaluators, cfg):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        bs_gt_mask,origin_size = [],None
        for data_sample in sampled_batch['data_samples']:
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            bs_gt_mask.append(gt_sem_seg)
            origin_size = data_sample.ori_shape

        bs_gt_mask = torch.cat(bs_gt_mask, dim = 0).to(device)
        outputs = model.forward_coarse(sampled_batch, cfg.val_prompt_type)
        logits = outputs['logits_256']  # (bs or k_prompt, 1, 256, 256)
        bs_pred_mask = (logits>0).detach()
        bs_pred_origin = F.interpolate(bs_pred_mask.type(torch.uint8), origin_size, mode="nearest")

        all_datainfo = []
        for idx,datainfo in enumerate(sampled_batch['data_samples']):
            datainfo.pred_sem_seg = PixelData(sem_seg=bs_pred_origin[idx])
            pred_logits = logits[idx][0].detach().cpu()
            pred_inst_mask = evaluators['inst_evaluator'].find_connect(pred_logits)
            panoptic_seg = torch.full((256, 256), 0, dtype=torch.int32, device=device)
            if np.sum(pred_inst_mask) > 0:
                iid = np.unique(pred_inst_mask)
                for i in iid[1:]:
                    mask = pred_inst_mask == i
                    panoptic_seg[mask] = (1 + i * INSTANCE_OFFSET)
            datainfo.pred_panoptic_seg = PixelData(sem_seg=panoptic_seg[None])
            all_datainfo.append(datainfo.to_dict())

            gt_inst_seg = datainfo.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask = onehot2instmask(gt_inst_seg)    # tensor, (h, w)
            
            evaluators['inst_evaluator'].process(gt_inst_mask, pred_logits)
        
        evaluators['semantic_evaluator'].process(None, all_datainfo)
        evaluators['panoptic_evaluator'].process(None, all_datainfo)
    
    total_datas = len(val_loader)*cfg.val_bs
    semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
    panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
    inst_metrics = evaluators['inst_evaluator'].evaluate(total_datas)
    metrics = dict(
        semantic = semantic_metrics,
        panoptic = panoptic_metrics,
        inst = inst_metrics,
    )
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
                use_embed = True,
                sam_ckpt = cfg.sam_ckpt,
                device = device
            ).to(device)
    
    # create logger
    logger,files_save_dir = get_logger(
        args.record_save_dir, model, cfg, logger_name)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    # get optimizer and lr_scheduler
    optimizer,lr_scheduler = get_train_strategy(model, cfg)
    # get evaluator
    evaluators = get_metrics(['pixel','inst'], metainfo, restinfo)

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
        best_score = metrics['inst']['AJI']
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
python scripts/panoptic/binary/coarse_main.py \
    configs/datasets/monuseg.py \
    --record_save_dir logs/coarse/train_all_ours/monuseg_p256
'''
