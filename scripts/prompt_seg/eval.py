import os
import torch
import argparse
from utils import set_seed, draw_pred,draw_multi_cls_pred
from models.prompt_seg import PromptSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
from utils.iou_metric import IoUMetric
import numpy as np
import torch.nn.functional as F
from utils.local_visualizer import SegLocalVisualizer

parser = argparse.ArgumentParser()

# base args
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--prompt_types', nargs='*', default=[], help='a list of prompt type,eg: box,point')

# about dataset
parser.add_argument('--dataset_domain', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--val_bs', type=int, default=1, help='validation batch_size per gpu')
parser.add_argument('--visual_pred', action='store_true')
parser.add_argument('--train_load_parts', nargs='*', type=int, default=[])
parser.add_argument('--val_load_parts', nargs='*', type=int, default=[])

# about model
parser.add_argument('--semantic_module_depth', type=int)
parser.add_argument('--use_inner_feat', action='store_true')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

args = parser.parse_args()

def val_one_epoch(model: PromptSegNet, val_loader):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        
        bs_mask_512 = sampled_batch['mask_512'].to(device) # shape: [bs, 512, 512]
        mask_1024 = sampled_batch['mask_1024'] # shape: [bs, 1024, 1024]
        bs_image_embedding,inter_feature = model.gene_img_embed(sampled_batch)
        if 'all_boxes' in args.prompt_types:
            mask_1024_binary = sampled_batch['mask_1024_binary'] # shape: [bs, 1024, 1024]
            # all_boxes_low_logits.shape: (k_boxes, num_cls, 512, 512)
            # all_prompts:list [{point, box, all_boxes},{}]
            all_boxes_low_logits, all_prompts = model.forward_single_class(bs_image_embedding,inter_feature, mask_1024_binary)
            # pred_logits.shape: (num_cls, 512, 512)
            bs_pred_logits,_ = torch.max(all_boxes_low_logits, dim=0)
            bs_pred_logits = bs_pred_logits.unsqueeze(0)    # (1, num_cls, 512, 512)
        else:
            all_cls_logits,all_cls_prompts = [],[]
            for cls_i in range(model.num_classes):
                mask_1024_cls_i = (mask_1024 == (1 if model.num_classes == 1 else cls_i)).to(torch.uint8)
                # low_logits.shape: (bs, num_cls, 512, 512)
                low_logits, prompts = model.forward_single_class(bs_image_embedding,inter_feature, mask_1024_cls_i)
                # cls_low_logits.shape: (bs, 1, 512, 512)
                cls_low_logits,_ = torch.max(low_logits, dim = 1, keepdim=True)
                all_cls_logits.append(cls_low_logits)
                all_cls_prompts.append(prompts)
            
            # bs_pred_logits.shape: (bs, num_cls, 512, 512)
            bs_pred_logits = torch.cat(all_cls_logits, dim=1)
        
        if model.num_classes == 1:
            bs_pred_mask = (bs_pred_logits>0).detach()
        else:
            bs_pred_mask = bs_pred_logits.argmax(dim=1, keepdim=True).detach()

        for pred_mask, mask_512 in zip(bs_pred_mask, bs_mask_512):
            test_evaluator.process(pred_mask, mask_512)

        if args.visual_pred and i_batch % 50 == 0:
            all_cls_logits_1024 = F.interpolate(bs_pred_logits, (1024, 1024), mode="bilinear", align_corners=False)
            if model.num_classes == 1:
                pred_mask_1024 = (all_cls_logits_1024 > 0).detach()

                for idx_in_batch in range(args.val_bs):
                    # first 0 mean class idx, second 0 mean batch idx
                    # single_img_prompts: dict(point=[x1,y1], box=[x1,y1,x2,y2])
                    single_img_prompts = all_cls_prompts[0][idx_in_batch]
                    coords_torch = None
                    if single_img_prompts['point'] is not None:
                        coords_torch = torch.as_tensor(np.array([single_img_prompts['point']]), dtype=torch.float)
                    draw_pred(sampled_batch, idx_in_batch, pred_mask_1024[idx_in_batch], pred_save_dir, coords_torch, single_img_prompts['box'])
            else:
                pred_mask_1024 = all_cls_logits_1024.argmax(dim=1).detach()
                pred_mask_1024[mask_1024==255] = 255
                for idx_in_batch in range(args.val_bs):
                    coords_torch,boxes,all_cls_agnostic_boxes = None,[],None
                    if 'all_boxes' in args.prompt_types:
                        all_cls_agnostic_boxes = all_prompts[idx_in_batch]['all_boxes']
                    else:
                        points = [cls_i_prompt[idx_in_batch]['point'] for cls_i_prompt in all_cls_prompts]
                        points = list(filter(None, points)) # [[x1,y1],...,[x1,y1]]
                        boxes = [cls_i_prompt[idx_in_batch]['box'] for cls_i_prompt in all_cls_prompts]    # [[x1,y1,x2,y2],...,[x1,y1,x2,y2]]
                        coords_torch = None
                        if len(points) > 0:
                            coords_torch = torch.as_tensor(np.array(points), dtype=torch.float)
                    
                    draw_multi_cls_pred(
                        sampled_batch, idx_in_batch, pred_mask_1024[idx_in_batch], pred_save_dir, 
                        metainfo, coords_torch, boxes, all_cls_agnostic_boxes)
    
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
                use_embed = args.use_embed,
                sam_ckpt = sam_ckpt,
                prompt_types = args.prompt_types,
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
        seg_local_visualizer = SegLocalVisualizer(
            save_dir = pred_save_dir,
            classes = metainfo['classes'],
            palette = metainfo['palette'],
        )
    
    # get evaluator
    test_evaluator = IoUMetric(iou_metrics=['mIoU','mFscore'])
    test_evaluator.dataset_meta = metainfo

    # val
    model.load_parameters(args.ckpt_path)
    metrics = val_one_epoch(model, val_dataloader)
    del metrics['ret_metrics_class']
    print(str(metrics))


'''
python scripts/prompt_seg/eval.py \
    logs/prompt_seg/pannuke_binary/2024_06_03_01_07_26 \
    logs/prompt_seg/pannuke_binary/2024_06_03_01_07_26/checkpoints/best_miou.pth \
    --dataset_domain medical \
    --dataset_name pannuke_binary \
    --train_load_parts 1 2 \
    --val_load_parts 3 \
    --num_classes 1 \
    --use_embed \
    --use_inner_feat \
    --semantic_module_depth 1 \
    --prompt_types box point \
    --val_bs 16 \
    --visual_pred
'''
