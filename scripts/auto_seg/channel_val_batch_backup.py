import time
import os
import torch
import argparse
from utils import set_seed
from models.auto_seg import ChannelProjectorNet
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader
from utils.iou_metric import BinaryIoUScore
import matplotlib.pyplot as plt
from utils.iou_metric import IoUMetric
from utils.visualization import show_mask,show_points
import torch.nn.functional as F
import numpy as np

from models.sam.utils.amg import MaskData, batch_iterator

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str,
                    default='whu', help='max points select in one episode')
parser.add_argument('--dir_name', type=str)
parser.add_argument('--n_per_side', type=int, default=32)
parser.add_argument('--points_per_batch', type=int, default=64)
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
    model = ChannelProjectorNet(
        n_per_side = args.n_per_side,
        points_per_batch = args.points_per_batch,
        sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
    ).to(device)
    model.eval()
    
    # dataset_config = dict(
    #     whu = '/nfs/zly/datasets/WHU-Building',
    #     inria = 'datasets/InriaBuildingDataset'
    # )
    dataset_config = dict(
        whu = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building',
        inria = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'
    )
    # load datasets
    dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        mode = 'val',
        use_embed = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    test_evaluator = IoUMetric(iou_metrics=['mIoU'])
    test_evaluator.dataset_meta = dataset.METAINFO
    all_metrics = []
    all_miou = []
    max_iou,max_epoch = 0,0

    for epoch_num in range(1, args.max_epochs):
    # for epoch_num in range(4,5):
        pth_load_path = f'{record_save_dir}/checkpoints/epoch_{epoch_num}.pth'
        print(f'load pth from {pth_load_path}')
        model.load_parameters(pth_load_path)
        for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
            mask_256 = sampled_batch['mask_256'].to(device)
            gt_origin_masks = sampled_batch['mask_512']
            bs_image_embedding = sampled_batch['img_embed'].to(device)
            image_pe = model.prompt_encoder.get_dense_pe()
            all_segment_logits = []

            # point_mask = torch.ones((1024,1024), dtype=bool)
            for (point_batch,) in batch_iterator(model.points_per_batch, model.points_for_sam):
                # points_new = []
                # for point_i in point_batch:
                #     point_i = point_i.astype(int)
                #     if not point_mask[point_i[1],point_i[0]]:
                #         points_new.append(point_i)
                # if len(points_new) == 0:
                #     continue
                with torch.no_grad():
                    bs_coords_torch_every = torch.as_tensor(np.array(point_batch), dtype=torch.float, device=device).unsqueeze(1)
                    bs_labels_torch_every = torch.ones(bs_coords_torch_every.shape[:2], dtype=torch.int, device=device)
                    points_every = (bs_coords_torch_every, bs_labels_torch_every)
                    sparse_embeddings_every, dense_embeddings_every = model.prompt_encoder(
                        points=points_every,
                        boxes=None,
                        masks=None,
                    )

                    low_res_logits_bs, iou_pred_bs, embeddings_64_bs, embeddings_256_bs, mask_token_out_bs = model.mask_decoder(
                        image_embeddings = bs_image_embedding,
                        image_pe = image_pe,
                        sparse_prompt_embeddings = sparse_embeddings_every,
                        dense_prompt_embeddings = dense_embeddings_every,
                    )
                    pred_mask_256 = low_res_logits_bs.flatten(0, 1) > 0
                    batch_data = MaskData(
                        logits = low_res_logits_bs.flatten(0, 1),    # shape: (points_per_batch, 256, 256)
                        masks = pred_mask_256,
                        iou_preds = iou_pred_bs.flatten(0, 1),  # shape: (points_per_batch,)
                        seg_matched_gt = pred_mask_256 & mask_256,
                        embed_256 = embeddings_256_bs,  # shape: (points_per_batch, 32, 256, 256)
                        mask_token_out = mask_token_out_bs,  # shape: (points_per_batch, 1, 256)
                        points = torch.as_tensor(point_batch.repeat(low_res_logits_bs.shape[1], axis=0)), # shape: (points_per_batch, 2)
                    )

                outputs = model.forward_batch_points(batch_data)
                all_segment_logits.append(outputs['keep_cls_logits'].detach())
            
            # all_segment_logits.shape: (n_per_side**2, 1, 256, 256 )
            all_segment_logits = torch.cat(all_segment_logits, dim = 0)
            pred_logits,_ = torch.max(all_segment_logits, dim=0)
            pred_logits = F.interpolate(
                pred_logits.unsqueeze(0),
                (512, 512),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            pred_mask = (pred_logits>0).cpu()
            gt_origin_masks[gt_origin_masks == 0] = 255
            test_evaluator.process(pred_mask, gt_origin_masks)

        metrics = test_evaluator.evaluate(len(dataloader))
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
python scripts/projector/channel_val_batch_backup.py \
    --dir_name 2024_05_15_12_12_57 \
    --n_per_side 64 \
    --points_per_batch 256 \
    --max_epochs 9 \
    --dataset_name whu
'''
