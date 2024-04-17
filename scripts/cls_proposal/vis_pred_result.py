import time
import os
import torch
import argparse
from help_func.tools import set_seed
from models.cls_proposal import ClsProposalNet
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
# from datasets.loveda import LoveDADataset
from torch.utils.data import DataLoader
from utils.iou_metric import IoUMetric,BinaryIoUScore
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.visualization import show_mask,show_points

parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type=str)
parser.add_argument('--ckpt_name', type=str)
parser.add_argument('--dataset_name', type=str,
                    default='whu', help='max points select in one episode')
parser.add_argument('--use_module', type=str)
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--num_points', nargs='+', type=int)
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = f'logs/cls_proposal/{args.dir_name}'
    
    # register model
    model = ClsProposalNet(
                num_classes = args.num_classes,
                num_points = args.num_points,
                useModule = args.use_module
            ).to(device)
    model.eval()
    
    dataset_config = dict(
        whu = 'datasets/WHU-Building',
        inria = 'datasets/InriaBuildingDataset'
    )
    # load datasets
    dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        mode = 'val',
        use_embed = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 4,
        drop_last = True)
    
    test_evaluator = IoUMetric(iou_metrics=['mIoU'])
    test_evaluator.dataset_meta = dataset.METAINFO
    pth_load_path = f'{record_save_dir}/checkpoints/{args.ckpt_name}.pth'
    model.load_parameters(pth_load_path)
    for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
        mask_1024 = sampled_batch['mask_1024'].to(device)
        bs_image_embedding = sampled_batch['img_embed'].to(device)
        
        outputs = model(bs_image_embedding, mask_1024)

        pred_logits = outputs['low_res_masks']
        # shape: [num_classes, 1024, 1024]
        pred_logits = F.interpolate(
            pred_logits,
            (512, 512),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pred_mask = (pred_logits>0).detach().cpu()
        gt_origin_masks = sampled_batch['mask_512']
        gt_origin_masks[gt_origin_masks == 0] = 255
        test_evaluator.process(pred_mask, gt_origin_masks)
        
        if i_batch % 50 == 0:
            gt_origin_masks[gt_origin_masks == 255] = 0
            iou_score = BinaryIoUScore(pred_mask, gt_origin_masks).item()

            pred_save_dir = f'{record_save_dir}/pred_vis'
            os.makedirs(pred_save_dir, exist_ok=True)
            fig = plt.figure(figsize=(12,6))
            ax_0 = fig.add_subplot(121)
            origin_image = sampled_batch['origin_image'][0]
            ax_0.imshow(origin_image.permute(1,2,0).cpu().numpy())
            show_mask(gt_origin_masks[0].cpu(), ax_0)
            ax_0.set_title('GT mask')
        
            ax_1 = fig.add_subplot(122)
            ax_1.imshow(origin_image.permute(1,2,0).cpu().numpy())
            show_mask(pred_mask[0].numpy(), ax_1)
            # 当前图有前景区域，坐标都是非负值
            if torch.sum(outputs['points'] < 0) == 0:
                coords_torch = (outputs['points'][0]).cpu() // 2
                labels_torch = torch.ones(len(coords_torch))
                show_points(coords_torch, labels_torch, ax_1)
            ax_1.set_title(f'pred_iou: {iou_score:.3f}')

            plt.tight_layout()
            image_name = sampled_batch['meta_info']['img_name'][0]
            plt.savefig(f'{pred_save_dir}/{image_name}')
            plt.close()

    metrics = test_evaluator.evaluate(len(dataloader))
    print(str(metrics) + '\n')

'''
python scripts/cls_proposal/vis_pred_result.py \
    --dir_name whu-sota \
    --ckpt_name epoch_17\
    --num_points 1 0 \
    --use_module conv \
    --dataset_name whu

'''
