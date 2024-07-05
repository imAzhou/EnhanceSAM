import os
import torch
from torch import Tensor
import numpy as np
import argparse
from utils import set_seed, get_logger, get_train_strategy, calc_loss,show_mask,show_box,draw_semantic_pred,calc_cls_loss_fn
from models.two_stage_seg import TwoStageNet
from datasets.panoptic.create_loader import gene_loader_trainval
from mmengine.config import Config
from utils.metrics import get_metrics
from tqdm import tqdm
import torch.nn.functional as F
from mmengine.structures import PixelData
import json
from models.sam.utils.amg import MaskData, batch_iterator
from utils.sam_postprocess import process_batch,process_img_points,format2panoptic,format2AJI
from scipy.ndimage import binary_dilation
import time
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--sample_points', type=int, default=256)
parser.add_argument('--points_per_batch', type=int, default=256)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

def onehot2instmask(onehot_mask):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
    
    return instmask

def match_gt2sp(gt_mask: Tensor, salient_points):
    '''
    for each salient point match a gt mask
    Args:
        - gt_mask: tensor, shape is (h,w), value is instance id, 0 mean background.
        - salient_points: np.array, shape is (k,2), scale shoud be same with gt
    '''
    h,w = gt_mask.shape
    matched_masks = torch.zeros((len(salient_points), h, w), dtype=torch.int32)
    instance_id = gt_mask[salient_points[:, 1], salient_points[:, 0]]
    non_zero_idx = torch.nonzero(instance_id)

    for idx in non_zero_idx:
        matched_masks[idx] = (gt_mask == instance_id[idx]).int()
    return matched_masks

def match_gt2sp_bgequalone(gt_mask: Tensor, salient_points):
    '''
    for each salient point match a gt mask
    Args:
        - gt_mask: tensor, shape is (h,w), value is instance id, 0 mean background.
        - salient_points: np.array, shape is (k,2), scale shoud be same with gt
    '''
    h,w = gt_mask.shape
    matched_masks = torch.zeros((len(salient_points), h, w), dtype=torch.int32)
    matched_labels = torch.zeros((len(salient_points)), dtype=torch.int32)
    instance_id = gt_mask[salient_points[:, 1], salient_points[:, 0]]

    for idx,iid in enumerate(instance_id):
        matched_labels[idx] = (iid > 0).int()
        matched_masks[idx] = (gt_mask == iid).int()
    
    return matched_masks,matched_labels

def load_salient_points(data_sample):
    img_path = data_sample.img_path
    img_dir,img_name = os.path.dirname(img_path),os.path.basename(img_path)
    purename = img_name.split('.')[0]
    root_path = img_dir.replace('img_dir', 'salient_p')
    prestore_json_path = f'{root_path}/{purename}.json'
    with open(prestore_json_path, 'r') as json_f:
        salient_points = json.load(json_f)
        bs_points = salient_points['points']
    return np.array(bs_points),salient_points['scale_ratio']

def draw_panoptic_pred(sample_data, pred_results, save_dir):
    '''
    Args:
        - sample_data: DetDataSample
        - pred_results: dict which include 'bboxes' 'scores' 'masks', all in 256 scale.
    '''
    cell_color = [47, 243, 15]
    img_path = sample_data.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_mask = sample_data.gt_sem_seg.sem_seg
    gt_boxes = sample_data.gt_instances.bboxes.tensor
    boxes_clsids = sample_data.gt_instances.labels

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(121)
    ax.imshow(image)
    show_mask(gt_mask, ax, rgb = cell_color)
    for box,cls_id in zip(gt_boxes, boxes_clsids): 
        cls_color = cell_color
        edgecolor = np.array([cls_color[0]/255, cls_color[1]/255, cls_color[2]/255, 1])
        show_box(box, ax, edgecolor=edgecolor)
    ax.set_title('GT info')

    ax = fig.add_subplot(122)
    ax.imshow(image)
    show_mask(pred_results['masks'], ax, rgb = cell_color)
    for box,cls_id in zip(pred_results['bboxes'], pred_results['boxes_clsids']): 
        cls_color = cell_color
        edgecolor = np.array([cls_color[0]/255, cls_color[1]/255, cls_color[2]/255, 1])
        show_box(box, ax, edgecolor=edgecolor)
    ax.set_title('pred results')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{image_name}')
    plt.close()

def draw_point_mask(sample_data, b_points, b_gtmask, b_gt_cls, i_batch):
    b_points = b_points // 4
    cell_color = [47, 243, 15]
    img_path = sample_data.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_mask = sample_data.gt_sem_seg.sem_seg
    random_idx = np.random.randint(0, high=b_points.shape[0], size=3)
    s_p,s_m,s_c = b_points[random_idx][:,0,:],b_gtmask[random_idx],b_gt_cls[random_idx]

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(221)
    ax.imshow(image)
    show_mask(gt_mask, ax, rgb = cell_color)
    ax.scatter(b_points[:, 0, 0], b_points[:, 0, 1], color='green', marker='*', s=50, edgecolor='white', linewidth=1.25)
    ax.set_title('GT info')

    ax = fig.add_subplot(222)
    ax.imshow(image)
    show_mask(s_m[0], ax, rgb = cell_color)
    ax.scatter(s_p[0, 0], s_p[0, 1], color='green', marker='*', s=100, edgecolor='white', linewidth=1.25)
    ax.set_title(f'point cls: {s_c[0]}')

    ax = fig.add_subplot(223)
    ax.imshow(image)
    show_mask(s_m[1], ax, rgb = cell_color)
    ax.scatter(s_p[1, 0], s_p[1, 1], color='green', marker='*', s=100, edgecolor='white', linewidth=1.25)
    ax.set_title(f'point cls: {s_c[1]}')

    ax = fig.add_subplot(224)
    ax.imshow(image)
    show_mask(s_m[2], ax, rgb = cell_color)
    ax.scatter(s_p[2, 0], s_p[2, 1], color='green', marker='*', s=100, edgecolor='white', linewidth=1.25)
    ax.set_title(f'point cls: {s_c[2]}')
    
    plt.tight_layout()
    os.makedirs(f'visual_results/matched_mask_3', exist_ok=True)
    plt.savefig(f'visual_results/matched_mask_3/{i_batch}_pred_{image_name}')
    plt.close()

def sample_point_fn(proposal_points, sample_num, shuffle = True):
    '''
    proposal_points: (k,2) (x,y)
    '''
    points = np.array(proposal_points)

    if len(points) >= sample_num:
        sample_points = points[np.random.choice(len(points), sample_num, replace=False)]
    else:
        point_map = np.zeros((1024, 1024))  # (w,h)
        structure = np.ones((16, 16)) # 点所在的 16*16 patch 块都不作为候选点
        point_map[points[:, 0], points[:, 1]] = 1
        point_map = binary_dilation(point_map, structure=structure).astype(point_map.dtype)
        zero_points = np.argwhere(point_map == 0)
        supplement_num = sample_num - len(points)
        # selected_points (x,y)
        selected_points = zero_points[np.random.choice(len(zero_points), supplement_num, replace=False)]
        sample_points = np.concatenate([points, selected_points], axis=0)
    
    if shuffle:
        np.random.shuffle(sample_points)

    return sample_points

def train_one_epoch(model: TwoStageNet, train_loader, optimizer, logger, cfg):
        
    model.train()
    len_loader = len(train_loader)
    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        # if i_batch > 5:
        #     break
        # 过滤掉没有显著点的训练样本，不参与训练
        data_sample = sampled_batch['data_samples'][0]
        bs_points,scale_ratio = load_salient_points(data_sample)
        if len(bs_points) == 0:
            continue
        
        gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
        inst_mask = onehot2instmask(gt_inst_seg)    # tensor, (h, w)
        bs_points = sample_point_fn(bs_points, args.sample_points)
        
        if cfg.use_cls_predict:
            bs_gt_mask,bs_target_label = match_gt2sp_bgequalone(inst_mask, bs_points//scale_ratio)
        else:
            bs_gt_mask = match_gt2sp(inst_mask, bs_points//scale_ratio)    # tensor, (k_points, h, w)  with binary pixel value
        bs_points = bs_points[:, np.newaxis, :]     # np.array, (k, 1, 2)
    
        if bs_gt_mask.shape[-1] != 256:
            bs_gt_mask = F.interpolate(bs_gt_mask.float().unsqueeze(1), (256, 256), mode="nearest").squeeze(1)

        pb = args.points_per_batch
        batch_points = batch_iterator(pb, bs_points)
        batch_gt_mask = batch_iterator(pb, bs_gt_mask)
        batch_gt_class = batch_iterator(pb, bs_target_label)
        for (b_points,), (b_gt_mask,), (b_gt_cls,) in zip(batch_points, batch_gt_mask, batch_gt_class):
            # draw_point_mask(data_sample, b_points, b_gt_mask, b_gt_cls, i_batch)
            outputs = model.forward_fine(sampled_batch, b_points)
            logits = outputs['logits_256']  # (bs or k_prompt, 1, 256, 256)
            if cfg.use_cls_predict:
                # mask_loss = calc_loss(pred_logits=logits, target_masks=b_gt_mask.to(device), args=cfg)
                cls_logits = outputs['cls_pred']
                loss = calc_cls_loss_fn(cls_logits, b_gt_cls.to(device))
                # loss = mask_loss + cls_loss
                # pred_cls = ((F.sigmoid(cls_logits)) > 0.5).int().detach().squeeze(1).cpu()
                # draw_point_mask(data_sample, b_points, (logits>0).squeeze(1).cpu(),pred_cls, i_batch)
            else:
                loss = calc_loss(pred_logits=logits, target_masks=b_gt_mask.to(device), args=cfg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (i_batch+1) % args.print_interval == 0:
            logger.info(f'iteration {i_batch+1}/{len_loader}: loss: {loss.item():.6f}')
    
def val_one_epoch(model: TwoStageNet, val_loader, evaluators, cfg):
    
    model.eval()
    total_cls_acc = 0.
    iter_cnt = 0
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        data_sample = sampled_batch['data_samples'][0]
        bs_points,scale_ratio = load_salient_points(data_sample)    # np.array, (k_points, 2)
        gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
        
        inst_mask = onehot2instmask(gt_inst_seg)    # tensor, (h, w)
        bs_gt_mask,bs_target_label = match_gt2sp_bgequalone(inst_mask, bs_points//scale_ratio)
        
        _,h,w = gt_inst_seg.shape
        panoptic_seg = torch.full((h, w), 0, dtype=torch.int32, device=device)
        if len(bs_points) > 0:
            pb = args.points_per_batch
            batch_points = batch_iterator(pb, bs_points)
            batch_gt_class = batch_iterator(pb, bs_target_label)
            data = MaskData()
            for (b_points,), (b_gt_cls,) in zip(batch_points, batch_gt_class):
                b_points = b_points[:, np.newaxis, :]     # np.array, (k, 1, 2)
                with torch.no_grad():
                    outputs = model.forward_fine(sampled_batch, b_points)
                logits = outputs['logits_256'].detach().cpu()  # (k_prompt, 1, 256, 256)
                iou_pred = outputs['iou_pred'].detach().cpu()   # (k_prompt, 1)
                
                if cfg.use_cls_predict:
                    cls_preds = F.sigmoid(outputs['cls_pred'].detach().cpu())   # (k_prompt, 1)
                    cls_acc = torch.sum(b_gt_cls == (cls_preds.squeeze(1)>0.5)) / cls_preds.shape[0]
                    total_cls_acc += cls_acc
                    iter_cnt += 1
                    # print(f'pred class acc: {cls_acc.item():.2f}')
                else:
                    cls_preds = torch.ones_like(iou_pred)  # (k_prompt, 1)
                
                pred_cls_id = (cls_preds > 0.5).int().squeeze(1)
                draw_point_mask(data_sample, b_points, (logits>0).squeeze(1).cpu(), pred_cls_id, i_batch)
                pred_iou_thresh = 0.5
                pred_cls_thresh = 0.5
                stability_score_thresh = 0.7
                batch_data = process_batch(
                    b_points[:,0,:] // scale_ratio, logits, iou_pred, cls_preds,
                    orig_size = (h,w),
                    pred_iou_thresh = pred_iou_thresh,
                    pred_cls_thresh = pred_cls_thresh,
                    stability_score_thresh = stability_score_thresh
                )
                data.cat(batch_data)
                del batch_data
                
            data = process_img_points(data)
            
            # left_num = data['points'].shape[0]
            # print(f'inst mask num: {logits.shape[0]}, filter left: {left_num}')
            # masks = np.array(data["segmentations"])     # np.array, (k, h, w) 
            # if len(masks) > 0:
            #     merged_masks = np.max(masks, axis=0)     # np.array, (h, w) 
            #     draw_panoptic_pred(data_sample, {
            #         'bboxes':data['boxes'],
            #         'boxes_clsids': np.ones((data['boxes'].shape[0], ), dtype=np.int32),
            #         'masks': merged_masks,
            #     }, 'visual_results')
            
            panoptic_seg = format2panoptic(data, device)

        pan_results = PixelData(sem_seg=panoptic_seg[None])
        data_sample.pred_panoptic_seg = pan_results
        evaluators['panoptic_evaluator'].process(None, [data_sample.to_dict()])
        
        gt_map,pred_map = format2AJI(data_sample, panoptic_seg)
        # draw_semantic_pred(data_sample, torch.as_tensor(pred_map)>0, pred_save_dir='visual_results/for_aji')
        evaluators['AJI_evaluator'].process(gt_map,pred_map)

        bs_pred_mask = torch.as_tensor((pred_map>0).astype(int)).unsqueeze(0)
        data_sample.pred_sem_seg = PixelData(sem_seg=bs_pred_mask)
        evaluators['semantic_evaluator'].process(None, [data_sample.to_dict()])
    
    total_datas = len(val_loader)*cfg.val_bs
    if cfg.use_cls_predict:
        print(f'pred total class acc: {(total_cls_acc / iter_cnt).item():.2f}')

    semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
    # detection_metrics = evaluators['detection_evaluator'].evaluate(total_datas)
    panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
    aji_metrics = evaluators['AJI_evaluator'].evaluate(total_datas)
    
    return semantic_metrics,panoptic_metrics,aji_metrics


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
                split_self_attn = cfg.split_self_attn,
                use_cls_predict = cfg.use_cls_predict,
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
    evaluators = get_metrics(['pixel', 'inst'], metainfo, restinfo)

    # train and val in each epoch
    all_metrics,all_aji = [],[]
    max_aji,max_epoch = 0,0
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
        semantic_metrics,panoptic_metrics,aji_metrics = val_one_epoch(model, val_dataloader, evaluators, cfg)
        aji = aji_metrics['AJI']
        logger.info(f'epoch: {epoch_num} ' + str(aji_metrics))
        if aji > max_aji:
            max_aji = aji
            max_epoch = epoch_num
            save_mode_path = os.path.join(pth_save_dir, 'best_miou.pth')
            model.save_parameters(save_mode_path)

        all_aji.append(aji)
        all_metrics.append(f'epoch: {epoch_num} ' + str(aji_metrics) + '\n')
    
    print(f'max_aji: {max_aji}, max_epoch: {max_epoch}')
    # save result file
    config_file = os.path.join(files_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_metrics)
        f.write(f'\max_aji: {max_aji}, max_epoch: {max_epoch}\n')
        f.write(str(all_aji))


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
python scripts/panoptic/binary/fine_main_inst.py \
    configs/datasets/monuseg.py \
    --record_save_dir logs/fine/point_cls/monuseg_p256 \
    --sample_points 256
'''
