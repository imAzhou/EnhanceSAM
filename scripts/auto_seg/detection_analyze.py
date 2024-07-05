import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy, calc_loss,get_det, get_connection
from models.auto_seg import AutoSegNet
from models.prompt_seg import PromptSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
import matplotlib.pyplot as plt
from utils.visualization import show_mask,show_points
from models.sam.utils.amg import batch_iterator
import torch.nn.functional as F
import numpy as np
import cv2

parser = argparse.ArgumentParser()

# base args
parser.add_argument('coarse_ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_epochs', type=int, default=12, help='maximum epoch number to train and val')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')


# about dataset
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_aug', action='store_true')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--train_sample_num', type=int, default=-1)
parser.add_argument('--train_load_parts', nargs='*', type=int, default=[])
parser.add_argument('--val_load_parts', nargs='*', type=int, default=[])

# about model
parser.add_argument('--points_per_batch', type=int, default=64)
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

# about train
# parser.add_argument('--loss_type', type=str)
# parser.add_argument('--base_lr', type=float, default=0.0001)
# parser.add_argument('--warmup_epoch', default=6, type=int)
# parser.add_argument('--gamma', default=0.5, type=float)
# parser.add_argument('--dice_param', type=float, default=0.8)
# parser.add_argument('--save_each_epoch', action='store_true')

args = parser.parse_args()

def get_point_segment(bianry_mask_np, points):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(bianry_mask_np, connectivity=8)
    segments = []
    for x,y in points:
        idx = labels[y,x]   # labels: (h,w)
        point_segmentent = (labels == idx).astype(np.uint8)
        segments.append(point_segmentent)
    
    return segments

def draw_sam_pred(sampled_batch, logits_256_binary, sam_pred_mask, sam_ious, points, point_segments, vis_save_dir, batch_idx):
    image_name = sampled_batch['meta_info'][0]['img_name']
    image_path = sampled_batch['meta_info'][0]['img_path']
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = sampled_batch['mask_256'][0]

    save_dir = f'{vis_save_dir}/{image_name}'
    os.makedirs(save_dir, exist_ok=True)

    idx = 0
    coords_torch, labels_torch = points
    for pred_mask, iou, point,label, segment in zip(sam_pred_mask, sam_ious, coords_torch, labels_torch, point_segments):
        point = point.unsqueeze(0).cpu()
        label = label.unsqueeze(0).cpu()
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(221)
        ax.imshow(image)
        show_mask(gt.cpu(), ax)
        show_points(point, label, ax)
        ax.set_title('gt entirety mask')

        ax = fig.add_subplot(222)
        ax.imshow(image)
        show_mask(logits_256_binary, ax)
        show_points(point, label, ax)
        ax.set_title('stage1 pred mask')

        ax = fig.add_subplot(223)
        ax.imshow(image)
        show_mask(segment, ax)
        # show_points(point, label, ax)
        ax.set_title('point counter-segment')

        ax = fig.add_subplot(224)
        ax.imshow(image)
        show_mask(pred_mask.cpu(), ax)
        # show_points(point, label, ax)
        ax.set_title(f'sam output, iou: {iou.item():.2f}')

        filename = batch_idx * args.points_per_batch + idx
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{filename}.png')
        plt.close()
        idx += 1

def vis_one_epoch(model: AutoSegNet, prompt_model:PromptSegNet, train_loader, vis_save_dir):
    
    model.train()
    len_loader = len(train_loader)
    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        gt_mask_256 = sampled_batch['mask_256'].to(device)
        bs_image_embedding = model.gene_img_embed(sampled_batch)

        outputs = prompt_model.forward_single_class(sampled_batch, 'random_bbox')
        logits_256 = outputs['logits_256']  # logits_256.shape: (bs, 1, 256, 256)
        logits_256_gray = logits_256.detach().cpu().squeeze(0).squeeze(0)
        logits_256_gray[logits_256_gray<0] = 0
        min_v, max_v = torch.min(logits_256_gray),torch.max(logits_256_gray)
        if max_v == 0:
            continue
        
        logits_256_gray = ((logits_256_gray - min_v) / (max_v - min_v)) * 255
        logits_256_gray = logits_256_gray.numpy()
        edge_intensity = get_det(logits_256_gray)   # 每张耗时 1.2s 左右
        edge_intensity[edge_intensity>0] = 1
        avgidx, avgbboxs = get_connection(edge_intensity,thresh=1)
        
        valued_center = []
        for avgbox in avgbboxs:
            hmin,wmin,hmax,wmax = avgbox
            center = [int((hmax+hmin)/2),int((wmax+wmin)/2)]
            if logits_256_gray[center[0], center[1]] >= 60:
                valued_center.append([center[1], center[0]])
        logits_256_binary = (logits_256_gray > 0).astype(np.uint8)
        point_segments = get_point_segment(logits_256_binary, valued_center)

        pb = model.points_per_batch
        batch_points = batch_iterator(pb, valued_center)
        batch_segments = batch_iterator(pb, point_segments)

        batch_idx = 0
        for (points,),(point_segments,) in zip(batch_points, batch_segments):
            scale_ratio = 1024 // 256
            points = np.array(points) * scale_ratio
            coords_torch = torch.as_tensor(np.array(points), dtype=torch.float, device=device)
            labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
            point_prompt = (coords_torch, labels_torch)

            mask_prompt = torch.as_tensor(np.array(point_segments)).unsqueeze(1).float().to(device)
            
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=point_prompt,
                # points=None,
                boxes=None,
                masks=mask_prompt,
                # masks=None,
            )
            sam_outputs = model.mask_decoder(
                image_embeddings = bs_image_embedding,
                image_pe = model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
                multimask_output = True
            )
            sam_logits,sam_ious = sam_outputs[0],sam_outputs[1]
            sam_pred_mask = sam_logits > 0  # (bs, m, 256, 256)
            sam_pred_mask = sam_pred_mask[:, 0, ::].unsqueeze(1)
            sam_ious = sam_ious[:, 0].unsqueeze(1)

            coords_torch = coords_torch / scale_ratio
            point_prompt = (coords_torch, labels_torch)
            draw_sam_pred(
                sampled_batch, logits_256_binary, sam_pred_mask, sam_ious, point_prompt, point_segments, vis_save_dir, batch_idx)
            batch_idx +=1

        # pb = model.points_per_batch
        # for (batch_idx, (point_batch,)) in enumerate(batch_iterator(pb, ps)):
        #     outputs = model.forward_batch_points(bs_image_embedding, point_batch)
        #     pred_cls_logits = outputs['cls_logits']    # shape: (points_per_batch, 1, 256, 256)
        #     pred_sam_logits = outputs['sam_logits']
        #     sam_pred_mask_256 = pred_sam_logits.flatten(0, 1) > 0
        #     seg_gt = sam_pred_mask_256 & gt_mask_256   # shape: (points_per_batch, 256, 256)
            
        #     loss = calc_loss(pred_logits=pred_cls_logits, target_masks=seg_gt, args=args)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        # if (i_batch+1) % 20 == 0:
        #     logger.info(f'iteration {i_batch+1}/{len_loader}: loss: {loss.item():.6f}')


def main():
    # register model
    sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
    model = AutoSegNet(
        num_classes = args.num_classes,
        n_per_side = 32,
        points_per_batch = args.points_per_batch,
        use_embed = args.use_embed,
        sam_ckpt = sam_ckpt,
        device = device
    ).to(device)
    prompt_model = PromptSegNet(
        num_classes = 1,
        sm_depth = 2,
        use_inner_feat = True,
        use_embed = True,
        sam_ckpt = sam_ckpt,
        device = device
    ).to(device)
    prompt_model.load_parameters(args.coarse_ckpt_path)
    prompt_model.eval()

    vis_save_dir = f'{record_save_dir}/74_02/vis_point_segment_prompt_smallest'
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # data loader
    train_loader,val_dataloader,metainfo = gene_loader(
        data_tag = args.dataset_name,
        use_aug = args.use_aug,
        use_embed = args.use_embed,
        use_inner_feat = True,
        train_sample_num = args.train_sample_num,
        train_bs = 1,
        val_bs = 1,
        train_load_parts = args.train_load_parts,
        val_load_parts = args.val_load_parts,
    )
   
    vis_one_epoch(model, prompt_model, train_loader, vis_save_dir)
    


if __name__ == "__main__":

    device = torch.device(args.device)
    record_save_dir = f'logs/auto_seg/{args.dataset_name}'
    set_seed(args.seed)
    main()
    
    

'''
python scripts/auto_seg/main.py \
    --max_epochs 10 \
    --dataset_name inria \
    --n_per_side 64 \
    --points_per_batch 256 \
    --loss_type bce_bdice \
    --base_lr 0.0001 \
    --warmup_epoch 8 \
    --use_embed \
    --train_sample_num 900 \
    --save_each_epoch
    --device cuda:1
    --use_aug \
    --debug_mode
'''
