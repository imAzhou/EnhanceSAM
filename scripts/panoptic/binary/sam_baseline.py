import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy, calc_loss
from datasets.panoptic.create_loader import gene_loader_trainval
from mmengine.config import Config
from utils.metrics import get_metrics
from mmdet.evaluation.functional import INSTANCE_OFFSET
from mmengine.structures import PixelData
from tqdm import tqdm
import numpy as np
import cv2
import torch.nn.functional as F
from models.two_stage_seg import SAMBaselineNet
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--vis_result_save_dir', type=str)
parser.add_argument('--metrics', nargs='*', type=str, default=['pixel', 'inst'])
parser.add_argument('--prompt_type', nargs='*', type=str, default=['point', 'box'])
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual_pred', action='store_true')
parser.add_argument('--update_decoder', action='store_true')

args = parser.parse_args()

semantic_sets = ['whu', 'inria', 'mass']
instance_sets = ['pannuke_binary', 'monuseg', 'cpm17']

def onehot2instmask(onehot_mask):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
    
    return instmask

def remap_mask(inst_mask):
    remap_inst_mask = np.zeros_like(inst_mask)
    iid = np.unique(inst_mask)
    for i,idx in enumerate(iid[1:]):
        remap_inst_mask[inst_mask == idx] = i+1
    return remap_inst_mask

def find_connect(image_mask_np):
    '''
    Args:
        image_mask_np(numpy): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        [[x1,y1,x2,y2], ..., [x1,y1,x2,y2]]
    '''
    # 寻找连通域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=8)
    trans_box = lambda x1,y1,w,h: [x1,y1, x1 + w, y1 + h]
    # stats[0] 是背景框
    all_boxes = np.array([trans_box(x1,y1,w,h) for x1,y1,w,h,_ in stats[1:]])
    return torch.as_tensor(all_boxes), torch.as_tensor(labels), torch.as_tensor(centroids)

def draw_visual_result(pred_sema_mask, pred_inst_mask, is_semantic, image_name):
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    if is_semantic:
        ax.imshow(pred_sema_mask, cmap='gray')
    else:
        h,w = pred_inst_mask.shape
        show_color_gt = np.zeros((h,w,4))
        show_color_gt[:,:,3] = 1
        inst_nums = len(np.unique(pred_inst_mask)) - 1
        for i in range(inst_nums):
            color_mask = np.concatenate([np.random.random(3), [1]])
            show_color_gt[pred_inst_mask==i+1] = color_mask
        ax.imshow(show_color_gt)
    
    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def eval_sam():
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols=70)):

        data_sample = sampled_batch['data_samples'][0]
        origin_h,origin_w = sampled_batch['data_samples'][0].ori_shape
        if cfg.dataset_tag in semantic_sets:
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            gt_bboxes,gt_inst_mask,gt_centroids = find_connect(gt_sem_seg[0].to(torch.int8).numpy())
        else:
            gt_bboxes = data_sample.gt_instances.bboxes.tensor  # tensor, (inst_nums, 4)
            gt_centroids = data_sample.gt_instances.bboxes.centers
            gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask = onehot2instmask(gt_inst_seg)  # tensor, (h, w)   
        
        panoptic_seg = torch.full((origin_h,origin_w), 0, dtype=torch.int32, device=device)
        pred_sema_mask = torch.full((origin_h,origin_w), 0, dtype=torch.int8, device=device)
        pred_inst_mask = np.zeros((origin_h,origin_w), dtype=int)
        if len(gt_bboxes) > 0:
            scale = 1024 // origin_h
            # box_prompt: (k,1,4)
            box_prompt = torch.as_tensor(
                gt_bboxes*scale, dtype=torch.float, device=device).unsqueeze(1)
            point_prompt = (
                (gt_centroids*scale).unsqueeze(1).to(device),
                torch.ones(len(gt_centroids), dtype=torch.int, device=device).unsqueeze(1)
            )
            prompt_dict = dict(point_prompt=None,box_prompt=None)
            if 'point' in args.prompt_type:
                prompt_dict['point_prompt'] = point_prompt
            if 'box' in args.prompt_type:
                prompt_dict['box_prompt'] = box_prompt
            outputs = model(sampled_batch, prompt_dict)
            # pred_mask.shape: (box_nums, h, w)
            pred_mask = (outputs['logits_origin'].detach() > 0).squeeze(1)
            for iid, inst_mask in enumerate(pred_mask):
                iid += 1
                panoptic_seg[inst_mask] = (1 + iid * INSTANCE_OFFSET)
                pred_inst_mask[inst_mask.cpu().numpy()] = iid
                pred_sema_mask[inst_mask] = 1
            
            if args.visual_pred:
                img_path = data_sample.img_path
                image_name = img_path.split('/')[-1]
                draw_visual_result(pred_sema_mask.cpu(), pred_inst_mask, 
                                   cfg.dataset_tag in semantic_sets, image_name)
            
        data_sample.pred_panoptic_seg = PixelData(sem_seg = panoptic_seg[None])
        data_sample.pred_sem_seg = PixelData(sem_seg = pred_sema_mask[None])

        evaluators['semantic_evaluator'].process(None, [data_sample.to_dict()])
        evaluators['panoptic_evaluator'].process(None, [data_sample.to_dict()])
        pred_inst_mask = remap_mask(pred_inst_mask)
        evaluators['inst_evaluator'].process(gt_inst_mask, pred_inst_mask)
    
    total_datas = len(val_dataloader)*cfg.val_bs
    semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
    panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
    inst_metrics = evaluators['inst_evaluator'].evaluate(total_datas)
    metrics = dict(
        semantic = semantic_metrics,
        panoptic = panoptic_metrics,
        inst = inst_metrics,
    )
    return metrics

def train_sam(optimizer, logger, cfg):

    model.train()
    len_loader = len(train_dataloader)

    for i_batch, sampled_batch in enumerate(tqdm(train_dataloader, ncols=70)):
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
        loss = bce_dice_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i_batch+1) % args.print_interval == 0:
            logger.info(f'iteration {i_batch+1}/{len_loader}, bce_dice_loss: {bce_dice_loss.item():.6f}')


def main(logger_name, cfg):
    
    # create logger
    logger,files_save_dir = get_logger(
        args.record_save_dir, model, cfg, logger_name)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    # get optimizer and lr_scheduler
    optimizer,lr_scheduler = get_train_strategy(model, cfg)

    # train and val in each epoch
    all_metrics,all_value = [],[]
    max_value,max_epoch = 0,0
    for epoch_num in range(cfg.max_epochs):
        # train
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"epoch: {epoch_num}, learning rate: {current_lr:.6f}")
        train_sam(optimizer, logger, cfg)
        lr_scheduler.step()
        if cfg.save_each_epoch:
            save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
            model.save_parameters(save_mode_path)
            logger.info(f"save model to {save_mode_path}")

        # val
        metrics = eval_sam()
        best_score = metrics['inst']['PQ']
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
    
     # register model
    model = SAMBaselineNet(
        update_decoder = args.update_decoder,
        use_embed = cfg.dataset.load_embed,
        sam_ckpt = cfg.sam_ckpt,
        sam_type = cfg.sam_type,
        device = device
    ).to(device)

    # load datasets
    train_dataloader, val_dataloader, metainfo, restinfo = gene_loader_trainval(
        dataset_config = d_cfg, seed = args.seed)

    # get evaluator
    evaluators = get_metrics(args.metrics, metainfo, restinfo)

    if args.visual_pred:
        pred_save_dir = args.vis_result_save_dir
        os.makedirs(pred_save_dir, exist_ok = True)

    if args.update_decoder:
        search_lr = cfg.get('search_lr', None)
        search_loss_type = cfg.get('search_loss_type', None)
        for loss_type in search_loss_type:
            cfg.loss_type = loss_type
            for idx,base_lr in enumerate(search_lr):
                cfg.base_lr = base_lr
                set_seed(args.seed)
                logger_name = f'lr_{loss_type}_{idx}'
                main(logger_name, cfg)
    else:
        eval_sam()


'''
python scripts/panoptic/binary/sam_baseline.py \
configs/datasets/cpm17.py \
    --record_save_dir logs/debug \
    --prompt_type point \
    --vis_result_save_dir visual_results/sam_baseline_output/cpm17 \
    --visual_pred \
    --update_decoder
'''