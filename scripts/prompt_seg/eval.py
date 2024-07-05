import os
import torch
import argparse
from utils import set_seed, draw_pred, draw_multi_cls_pred
from models.prompt_seg import PromptSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
from utils.iou_metric import IoUMetric
import numpy as np
import cv2
import torch.nn.functional as F

parser = argparse.ArgumentParser()

# base args
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--prompt_type', type=str,)

# about dataset
parser.add_argument('--dataset_domain', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--val_bs', type=int, default=1, help='validation batch_size per gpu')
parser.add_argument('--visual_pred', action='store_true')
parser.add_argument('--save_result', action='store_true')
parser.add_argument('--train_load_parts', nargs='*', type=int, default=[])
parser.add_argument('--val_load_parts', nargs='*', type=int, default=[])
parser.add_argument('--bg_clsid', type=int, default=5)

# about model
parser.add_argument('--semantic_module_depth', type=int)
parser.add_argument('--use_inner_feat', action='store_true')
parser.add_argument('--use_multi_mlps', action='store_true')
parser.add_argument('--use_mask_prompt', action='store_true')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

args = parser.parse_args()

def val_one_epoch_multi(model: PromptSegNet, val_loader):
    
    model.eval()
    total_boxes,right_cls_boxes = 0,0
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        mask_512 = sampled_batch['mask_512'].to(device) # shape: [1, 512, 512]
        mask_1024 = sampled_batch['mask_1024']
        gt_boxes = sampled_batch['gt_boxes'][0]
        pred_mask_512 = torch.ones_like(mask_512).long() * args.bg_clsid

        if len(gt_boxes.keys()) > 0:
            mask_256_binary = sampled_batch['mask_256_binary'].to(device)
            outputs = model.forward_multi_class(sampled_batch, args.prompt_type,mask_256_binary)
            pred_mask_logits = outputs['pred_mask_logits']    # shape: (k, 1, 512, 512)
            pred_cls_logits = outputs['pred_cls_logits']   # shape: (k, num_class)
            target_masks = outputs['target_masks']    # shape: (k, 512, 512)
            target_cls = outputs['target_cls']   # shape: (k,)

            max_value, max_indices = torch.max(pred_mask_logits, dim=0)
            max_indices[max_value<=0] = -1
            pred_clsid = torch.argmax(pred_cls_logits, dim=1)   # shape: (k,)
            for kid,clsid in enumerate(pred_clsid):
                pred_mask_512[max_indices == kid] = clsid
                
            total_boxes += len(target_cls)
            right_cls_boxes += torch.sum(pred_clsid == target_cls)

        test_evaluator.process(pred_mask_512, mask_512)
        

        if args.visual_pred and i_batch % 50 == 0:
            pred_mask_1024 = F.interpolate(pred_mask_512.type(torch.uint8).unsqueeze(1), (1024, 1024), mode="nearest").squeeze(0)
            pred_mask_1024[mask_1024==255] = 255

            prompts = outputs['prompts']
            boxes,points = prompts['boxes'],prompts['points']
            if boxes is not None:
                boxes = boxes.squeeze(1).cpu()    # (k,4)
                boxes_clsids = pred_clsid.tolist()
            if points is not None:
                points = points[0].squeeze(0).cpu()   # (k,2)

            if len(gt_boxes.keys()) == 0:
                boxes,points = None, None
            
            draw_multi_cls_pred(
                sampled_batch, 0, pred_mask_1024[0], pred_save_dir, 
                metainfo, points, boxes, boxes_clsids)
    
    metrics = test_evaluator.evaluate(len(val_loader)*args.val_bs)
    metrics.update({
        'precision_box_cls': f'{(right_cls_boxes / total_boxes):.4f}',
    })
    return metrics


def val_one_epoch_binary(model: PromptSegNet, val_loader):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):

        bs_mask_512 = sampled_batch['mask_512'].to(device) # shape: [bs, 512, 512]
        outputs = model.forward_single_class(sampled_batch, args.prompt_type)
        logits_512 = outputs['logits_512']  # logits_512.shape: (bs, 1, 512, 512)
        logits_256 = outputs['logits_256']  # logits_256.shape: (bs, 1, 256, 256)
        bs_pred_mask_512 = (logits_512>0).type(torch.uint8).detach()
        bs_pred_mask_256 = (logits_256>0).type(torch.uint8).detach()
        bs = bs_mask_512.shape[0]
        for idx in range(bs):
            pred_mask_512 = bs_pred_mask_512[idx]
            gt_mask_512 = bs_mask_512[idx]
            pred_mask_256 = bs_pred_mask_256[idx]
            meta_info = sampled_batch['meta_info'][idx]

            test_evaluator.process(pred_mask_512, gt_mask_512)

            if args.save_result:
                img_name = meta_info['img_name']
                pred_mask_256 = pred_mask_256.squeeze(0).detach().cpu().numpy()
                cv2.imwrite(f'{result_dir}/{img_name}.png', pred_mask_256)
            
            if args.visual_pred and i_batch % 50 == 0:
                pred_mask_1024 = F.interpolate(pred_mask_512.unsqueeze(0), (1024, 1024), mode="nearest").squeeze(0)
                prompts = outputs['prompts'][idx]
                boxes,points = prompts['boxes'],prompts['points']
                if boxes is not None:
                    boxes = boxes.squeeze(1).cpu()   # (k,4)
                if points is not None:
                    points = points[0].squeeze(0).cpu()   # (k,2)

                vis_mask = pred_mask_1024.cpu()
                draw_pred(sampled_batch, idx, vis_mask, pred_save_dir, points, boxes)
    
    metrics = test_evaluator.evaluate(len(val_loader)*args.val_bs)
    return metrics

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # register model
    sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
    model = PromptSegNet(
                num_classes = args.num_classes,
                sm_depth = args.semantic_module_depth,
                use_inner_feat = args.use_inner_feat,
                use_multi_mlps = args.use_multi_mlps,
                use_mask_prompt = args.use_mask_prompt,
                use_embed = args.use_embed,
                sam_ckpt = sam_ckpt,
                device = device
            ).to(device)
    # data loader
    train_loader,val_dataloader,metainfo = gene_loader(
        dataset_domain = args.dataset_domain,
        data_tag = args.dataset_name,
        use_aug = False,
        use_embed = args.use_embed,
        use_inner_feat = args.use_inner_feat,
        train_sample_num = -1,
        train_bs = 1,
        val_bs = args.val_bs,
        train_load_parts = args.train_load_parts,
        val_load_parts = args.val_load_parts,
    )
    if args.visual_pred:
        pred_save_dir = f'{args.result_save_dir}/vis_pred'
        os.makedirs(pred_save_dir, exist_ok = True)
    if args.save_result:
        result_dir = f'{args.result_save_dir}/results'
        os.makedirs(result_dir, exist_ok = True)
    
    # get evaluator
    test_evaluator = IoUMetric(iou_metrics=['mIoU','mFscore'])
    test_evaluator.dataset_meta = metainfo

    # val
    model.load_parameters(args.ckpt_path)
    if args.num_classes > 1:
        metrics = val_one_epoch_multi(model, val_dataloader)
    else:
        metrics = val_one_epoch_binary(model, val_dataloader)
    del metrics['ret_metrics_class']
    print(str(metrics))


'''
python scripts/prompt_seg/eval.py \
    logs/prompt_seg/pannuke_binary/74_02 \
    logs/prompt_seg/pannuke_binary/74_02/checkpoints/best_miou.pth \
    --dataset_domain medical \
    --dataset_name pannuke_binary \
    --num_classes 1 \
    --use_embed \
    --use_inner_feat \
    --semantic_module_depth 2 \
    --prompt_type random_bbox \
    --val_bs 16 \
    --train_load_parts 1 2 \
    --val_load_parts 3 \
    --visual_pred
'''
