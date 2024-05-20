import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy,draw_pred
from models.prompt_seg import PromptSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
from utils.iou_metric import IoUMetric
import numpy as np

parser = argparse.ArgumentParser()

# base args
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')

# about dataset
parser.add_argument('--server_name', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--visual_pred', action='store_true')

# about model
parser.add_argument('--semantic_module', type=str)
parser.add_argument('--use_inner_feat', action='store_true')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

args = parser.parse_args()

def val_one_epoch(model: PromptSegNet, val_loader):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        
        mask_512 = sampled_batch['mask_512'].to(device) # shape: [bs, 512, 512]
        outputs = model(sampled_batch)
        # shape: [num_classes, 1024, 1024]
        pred_logits = outputs['pred_mask_512'].squeeze(0)
        pred_mask = (pred_logits>0).detach()

        if args.visual_pred and i_batch % 50 == 0:
            pred_mask_1024 = (outputs['pred_mask_1024'].squeeze(0) > 0).detach()
            point_box = outputs['bs_point_box'][0]
            coords_torch = None
            if point_box['point'] is not None:
                coords_torch = torch.as_tensor(np.array([point_box['point']]), dtype=torch.float)
            draw_pred(sampled_batch, pred_mask_1024[0], pred_save_dir, coords_torch, point_box['box'])

        # mask_512[mask_512 == 0] = 255
        test_evaluator.process(pred_mask, mask_512)
        
    metrics = test_evaluator.evaluate(len(val_loader))
    return metrics

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # register model
    sam_ckpt = dict(
        zucc = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth',
        hz = 'checkpoints_sam/sam_vit_h_4b8939.pth'
    )
    model = PromptSegNet(
                num_classes = args.num_classes,
                useModule = args.semantic_module,
                use_inner_feat = args.use_inner_feat,
                use_embed = args.use_embed,
                sam_ckpt = sam_ckpt[args.server_name],
                device = device
            ).to(device)
    # data loader
    train_loader,val_dataloader,metainfo = gene_loader(
        server_name = args.server_name,
        data_tag = args.dataset_name,
        use_aug = False,
        use_embed = args.use_embed,
        train_sample_num = -1,
        train_bs = 1,
        val_bs = 1
    )
    if args.visual_pred:
        pred_save_dir = f'{args.result_save_dir}/vis_pred'
        os.makedirs(pred_save_dir, exist_ok = True)
    
    # get evaluator
    test_evaluator = IoUMetric(iou_metrics=['mIoU','mFscore'])
    test_evaluator.dataset_meta = metainfo

    # val
    model.load_parameters(args.ckpt_path)
    metrics = val_one_epoch(model, val_dataloader)
    del metrics['ret_metrics_class']
    print(str(metrics))


'''
use_inner_feat
python scripts/prompt_seg/eval.py \
    best_results/prompt_seg/whu-400 \
    best_results/prompt_seg/whu-400/checkpoints/best_miou.pth \
    --server_name zucc \
    --dataset_name whu \
    --use_inner_feat \
    --semantic_module conv \
    --visual_pred
'''
