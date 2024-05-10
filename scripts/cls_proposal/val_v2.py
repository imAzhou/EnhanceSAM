import time
import os
import torch
import argparse
from models.cls_proposal_v2 import ClsProposalNet
from tqdm import tqdm
from utils import set_seed
# from datasets.loveda import LoveDADataset
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader
from utils.iou_metric import IoUMetric

parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type=str)
parser.add_argument('--use_module', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--num_points', nargs='+', type=int)
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--specify_epoch', type=int)
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual', action='store_true', help='If activated, the predict mask will be saved')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = f'logs/cls_proposal_v2/{args.dir_name}'
    
    # register model
    model = ClsProposalNet(
                num_classes = args.num_classes,
                num_points = args.num_points,
                useModule = args.use_module
            ).to(device)
    model.eval()
    
    # load datasets
    dataset_config = dict(
        whu = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building',
        inria = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'
    )
    dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        mode = 'val',
        # use_embed = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    
    test_evaluator = IoUMetric(iou_metrics=['mIoU'])
    test_evaluator.dataset_meta = dataset.METAINFO

    def eval_epoch(pth_load_path):
        model.load_parameters(pth_load_path)
        for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
            mask_1024 = sampled_batch['mask_1024'].to(device)
            bs_input_image = sampled_batch['input_image'].to(device)
            outputs = model(bs_input_image, mask_1024)

            # shape: [num_classes, 1024, 1024]
            pred_logits = outputs['pred_mask_512'].squeeze(0)

            pred_mask = (pred_logits>0).detach().cpu()
            gt_origin_masks = sampled_batch['mask_512']
            gt_origin_masks[gt_origin_masks == 0] = 255
            test_evaluator.process(pred_mask, gt_origin_masks)

        metrics = test_evaluator.evaluate(len(dataloader))
        return metrics
    
    if args.specify_epoch is not None:
        pth_load_path = f'{record_save_dir}/checkpoints/epoch_{args.specify_epoch}.pth'
        metrics = eval_epoch(pth_load_path)
        print(f'epoch: {args.specify_epoch} ' + str(metrics) + '\n')
    else:
        all_metrics = []
        all_miou = []
        max_iou,max_epoch = 0,0
        for epoch_num in range(args.max_epochs):
            pth_load_path = f'{record_save_dir}/checkpoints/epoch_{epoch_num}.pth'
            metrics = eval_epoch(pth_load_path)
            print(f'epoch: {epoch_num} ' + str(metrics) + '\n')
            all_miou.append(metrics['mIoU'])
            if metrics['mIoU'] > max_iou:
                max_iou = metrics['mIoU']
                max_epoch = epoch_num
            all_metrics.append(f'epoch: {epoch_num}' + str(metrics) + '\n')
        # save result file
        config_file = os.path.join(record_save_dir, 'pred_result.txt')
        with open(config_file, 'w') as f:
            f.writelines(all_metrics)
            f.write(f'\nmax_iou: {max_iou}, max_epoch: {max_epoch}\n')
            f.write(str(all_miou))

'''
python scripts/cls_proposal/val_v2.py \
    --dir_name 2024_05_09_15_16_03 \
    --batch_size 1 \
    --num_points 1 0 \
    --max_epochs 12 \
    --use_module both \
    --dataset_name whu
    --device cuda:1 \

        
python scripts/cls_proposal/val_v2.py \
    --dir_name sota-whu \
    --batch_size 1 \
    --num_points 1 0 \
    --specify_epoch 15 \
    --use_module conv \
    --dataset_name whu
    --device cuda:1 \
'''
