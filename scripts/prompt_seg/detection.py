import os
import torch
import argparse
from utils import set_seed,get_det, get_connection, draw_det_result
from models.prompt_seg import PromptSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
import numpy as np
import cv2
import time
from datetime import timedelta


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
thresh_center = 0.7
rad = 10

def val_one_epoch_binary(model: PromptSegNet, val_loader):
    
    model.eval()
    # all_img_bboxes = []
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):

        outputs = model.forward_single_class(sampled_batch, args.prompt_type)
        logits_256 = outputs['logits_256']  # logits_256.shape: (bs, 1, 256, 256)

        bs = logits_256.shape[0]
        for idx in range(bs):
            logits_256_gray = logits_256.detach().cpu()
            logits_256_gray = logits_256_gray[idx].squeeze(0)
            logits_256_gray[logits_256_gray<0] = 0
            min_v, max_v = torch.min(logits_256_gray),torch.max(logits_256_gray)
            if max_v> 0:
                logits_256_gray = ((logits_256_gray - min_v) / (max_v - min_v)) * 255
                logits_256_gray = logits_256_gray.numpy()
                edge_intensity = get_det(logits_256_gray)   # 每张耗时 1.2s 左右
                edge_intensity[edge_intensity>0] = 1
                avgidx, avgbboxs = get_connection(edge_intensity,thresh=1)
                
                pool = np.zeros((logits_256_gray.shape[0],logits_256_gray.shape[1]))
                valued_bboxes = []
                for avgbox in avgbboxs:
                    hmin,wmin,hmax,wmax = avgbox
                    center = [int((hmax+hmin)/2),int((wmax+wmin)/2)]
                    length = [hmax-hmin,wmax-wmin]
                    if logits_256_gray[center[0], center[1]] < 60:
                        continue
                    area = (hmax-hmin)*(wmax-wmin)
                    if area > 0:
                        valued_bboxes.append([wmin, hmin, wmax, hmax])
                    for i in range(max(0,center[0]-rad),min(logits_256_gray.shape[0],center[0]+rad)):
                        for j in range(max(0,center[1]-rad),min(logits_256_gray.shape[0],center[1]+rad)):
                            if logits_256_gray[i,j] >= thresh_center*logits_256_gray[center[0],center[1]]:
                                pool[i,j] = 255
                    
                
                draw_det_result(logits_256_gray, edge_intensity, valued_bboxes, pool, sampled_batch, det_save_dir)

    all_img_bboxes = np.array(all_img_bboxes)
    print(f'min: {np.min(all_img_bboxes)}, max: {np.max(all_img_bboxes)}')


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
    
    det_save_dir = f'{args.result_save_dir}/vis_detection'
    os.makedirs(det_save_dir, exist_ok = True)

    # val
    model.load_parameters(args.ckpt_path)
    val_one_epoch_binary(model, val_dataloader)



'''
python scripts/prompt_seg/detection.py \
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
    --val_load_parts 3
'''
