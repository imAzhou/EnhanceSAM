import os
import torch
import argparse
from utils import set_seed,show_multi_mask
from models.two_stage_seg import TwoStageNet
from datasets.panoptic.create_loader import gene_loader_eval
from mmengine.config import Config
from utils.metrics import get_metrics
from tqdm import tqdm
import numpy as np
from mmdet.evaluation.functional import INSTANCE_OFFSET
from mmengine.structures import PixelData
import matplotlib.pyplot as plt
import cv2
import copy
import random
from scipy.ndimage import binary_dilation, binary_erosion

parser = argparse.ArgumentParser()

# base args
parser.add_argument('config_file', type=str)
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--metrics', nargs='*', type=str, default=['pixel', 'inst'])
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual_pred', action='store_true')
parser.add_argument('--draw_simple', action='store_true')
parser.add_argument('--visual_interval', type=int, default=1)

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
    h,w = gt_mask.shape

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
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def draw_pred_simple(datainfo, gt_inst_mask, pred_inst_mask):
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

    fig = plt.figure(figsize=(12,6))
    
    ax = fig.add_subplot(121)
    inst_nums = len(np.unique(gt_inst_mask)) - 1
    copy_img = copy.deepcopy(img)
    for i in range(inst_nums):
        color_mask = [random.randint(0, 255) for _ in range(3)]
        copy_img[gt_inst_mask==i+1] = color_mask
    ax.imshow(copy_img)
    ax.set_axis_off()
    ax.set_title('color gt')

    ax = fig.add_subplot(122)
    inst_nums = len(np.unique(pred_inst_mask)) - 1
    copy_img = copy.deepcopy(img)
    for i in range(inst_nums):
        color_mask = [random.randint(0, 255) for _ in range(3)]
        copy_img[pred_inst_mask==i+1] = color_mask
    ax.imshow(copy_img)
    ax.set_axis_off()
    ax.set_title('color pred')
    
    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()


def val_one_epoch():
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols=70)):
        bs_gt_mask = []
        origin_h,origin_w = sampled_batch['data_samples'][0].ori_shape
        for data_sample in sampled_batch['data_samples']:
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            bs_gt_mask.append(gt_sem_seg)

        bs_gt_mask = torch.cat(bs_gt_mask, dim = 0).to(device)
        outputs = model.forward_coarse(sampled_batch)
        logits = outputs['logits_origin']  # (bs or k_prompt, 1, 256, 256)
        
        # bs_pred_mask = torch.argmax(logits, dim=1).detach()  # (bs or k_prompt, 256, 256)
        # bs_pred_origin = F.interpolate(
        #     bs_pred_mask.unsqueeze(1).type(torch.uint8), origin_size, mode="nearest").squeeze(1)
        
        # calc iou metrics
        all_datainfo = []
        for idx,datainfo in enumerate(sampled_batch['data_samples']):

            gt_inst_seg = datainfo.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask,gt_boundary_mask = onehot2instmask(gt_inst_seg,with_boundary=True)    # tensor, (h, w)
            gt_sem_seg = datainfo.gt_sem_seg.sem_seg.squeeze(0)    # tensor, (1, h, w) pixel value is cls id

            pred_cell_mask = (logits[idx,0,:,:].detach() > 0).type(torch.uint8)
            pred_boundary_mask = logits[idx,1,:,:].detach() > 0
            
            pred_cell_mask[pred_boundary_mask] = 0
            pred_inst_mask = evaluators['inst_evaluator'].find_connect(pred_cell_mask, dilation=True)
            evaluators['inst_evaluator'].process(gt_inst_mask, pred_inst_mask)
            
            panoptic_seg = torch.full((origin_h,origin_w), 0, dtype=torch.int32, device=device)
            if np.sum(pred_inst_mask) > 0:
                iid = np.unique(pred_inst_mask)
                for i in iid[1:]:
                    mask = pred_inst_mask == i
                    panoptic_seg[mask] = (1 + i * INSTANCE_OFFSET)
            datainfo.pred_panoptic_seg = PixelData(sem_seg=panoptic_seg[None])
            pred_cell_mask = (panoptic_seg > 0).type(torch.uint8)
            datainfo.pred_sem_seg = PixelData(sem_seg=pred_cell_mask[None])
            all_datainfo.append(datainfo.to_dict())

            if args.visual_pred and i_batch % args.visual_interval == 0:
                if args.draw_simple:
                    draw_pred_simple(datainfo,gt_inst_mask, pred_inst_mask)
                else:
                    draw_pred(datainfo, gt_sem_seg,pred_cell_mask, 
                          gt_inst_mask, pred_inst_mask, gt_boundary_mask, pred_boundary_mask.detach().cpu())
        
        evaluators['semantic_evaluator'].process(None, all_datainfo)
        evaluators['panoptic_evaluator'].process(None, all_datainfo)
    
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



if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    
    # load datasets
    val_dataloader, metainfo, restinfo = gene_loader_eval(
        dataset_config = cfg, seed = args.seed)

    # register model
    model = TwoStageNet(
        num_classes = cfg.num_classes,
        num_mask_tokens = cfg.num_mask_tokens,
        sm_depth = cfg.semantic_module_depth,
        use_inner_feat = cfg.use_inner_feat,
        use_boundary_head = cfg.use_boundary_head,
        use_embed = True,
        sam_ckpt = cfg.sam_ckpt,
        device = device
    ).to(device)
    
    if args.visual_pred:
        pred_save_dir = f'{args.result_save_dir}/vis_pred_simple'
        os.makedirs(pred_save_dir, exist_ok = True)

    # get evaluator
    evaluators = get_metrics(args.metrics, metainfo, restinfo)

    # val
    model.load_parameters(args.ckpt_path)
    metric_results = val_one_epoch()
    print(metric_results)


'''
python scripts/panoptic/binary/coarse_eval_2.py \
    logs/0_pannuke_binary_02/2024_08_04_18_21_29/config.py \
    logs/0_pannuke_binary_02/2024_08_04_18_21_29 \
    logs/0_pannuke_binary_02/2024_08_04_18_21_29/checkpoints/best.pth \
    --metrics pixel inst \
    --visual_pred \
    --draw_simple \
    --visual_interval 50 \
'''
