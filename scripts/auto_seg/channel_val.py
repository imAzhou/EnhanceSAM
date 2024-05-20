import time
import os
import torch
import argparse
from help_func.tools import set_seed
from models.auto_seg import ChannelProjectorNet
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader
from utils.iou_metric import BinaryIoUScore
import matplotlib.pyplot as plt
from utils.visualization import show_mask,show_points
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str,
                    default='whu', help='max points select in one episode')
parser.add_argument('--dir_name', type=str)
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--num_points', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--sam_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual', action='store_true', help='If activated, the predict mask will be saved')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = f'logs/projector/{args.dir_name}'
    
    # register model
    model = ChannelProjectorNet().to(device)
    model.eval()
    
    dataset_config = dict(
        whu = 'datasets/WHU-Building',
        inria = 'datasets/InriaBuildingDataset'
    )
    # load datasets
    dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        resize_size = args.img_size,
        mode = 'val',
        use_embed = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True)
    
    
    all_iou = []
    max_iou,max_epoch = 0,0
    for epoch_num in range(args.max_epochs):
    # for epoch_num in range(10,11):
        avarage_iou = 0.0
        pth_load_path = f'{record_save_dir}/checkpoints/epoch_{epoch_num}.pth'
        model.load_parameters(pth_load_path)
        for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
            gt_origin_masks = sampled_batch['mask_te_origin']
            image_batch, mask_256 = sampled_batch['image_tensor'].to(device), sampled_batch['mask_256'].to(device)
            
            bs_image_embedding = sampled_batch['img_embed'].to(device)
            outputs = model(bs_image_embedding, mask_256)

            pred_logits = outputs['keep_cls_logits']    # shape: (keep_by_nms, 1, 256, 256)
            pred_logits,_ = torch.max(pred_logits, dim=0)
            pred_logits = F.interpolate(
                pred_logits.unsqueeze(0),
                (512, 512),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            pred_mask = (pred_logits>0).detach().cpu()

            iou_score = BinaryIoUScore(pred_mask, gt_origin_masks).item()
            avarage_iou += iou_score

        avarage_iou /= len(dataloader)
        print(f'epoch: {epoch_num}, miou: {avarage_iou}')
        all_iou.append(f'epoch: {epoch_num}, miou: {avarage_iou}\n')
        if avarage_iou > max_iou:
            max_iou = avarage_iou
            max_epoch = epoch_num
    # save result file
    config_file = os.path.join(record_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_iou)
        f.write(f'\nmax_iou: {max_iou}, max_epoch: {max_epoch}\n')

'''
python scripts/projector/channel_val.py \
    --dir_name 2024_04_04_11_52_23 \
    --max_epochs 2
    --num_points 0 \
    --max_epochs 12 \
    --dataset_name whu
    --visual
    --max_epochs 12 \
    --visual

'''
