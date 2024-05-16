import time
import os
import torch
import argparse
from utils import set_seed
from models.projector import ChannelProjectorNet
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

def drwa_mask(image_name, img, gt, sam_seg, credible_p, credible_n, coords_torch):
    '''
    绘制四宫格，从上至下从左至右分别是：
        整图GT，选点及sam分割的块，
        可信前景区域，可信背景区域
    Args:
        img: np array, shape is (h,w,c)
        gt, sam_seg, credible_p, credible_n: tensor, shape is (h,w), value is 0/1
        coords_torch: (num_points, 2), 2 is (x,y)
    '''
    pred_save_dir = 'visual_result/projector'
    os.makedirs(pred_save_dir, exist_ok=True)
    labels_torch = torch.ones(len(coords_torch))

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(221)
    ax.imshow(img)
    show_mask(gt.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('pred mask')

    ax = fig.add_subplot(222)
    ax.imshow(img)
    show_mask(sam_seg.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('SAM seg')

    ax = fig.add_subplot(223)
    ax.imshow(img)
    show_mask(credible_p.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('credible positive region')

    ax = fig.add_subplot(224)
    ax.imshow(img)
    show_mask(credible_n.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('credible negative region')

    plt.tight_layout()
    
    plt.savefig(f'{pred_save_dir}/07_{image_name}')
    plt.close()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = f'logs/projector/{args.dir_name}'
    
    # register model
    model = ChannelProjectorNet(
        n_per_side = args.n_per_side,
        points_per_batch = args.points_per_batch,
    ).to(device)
    model.eval()
    
    dataset_config = dict(
        whu = '/nfs/zly/datasets/WHU-Building',
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    test_evaluator = IoUMetric(iou_metrics=['mIoU'])
    test_evaluator.dataset_meta = dataset.METAINFO
    all_metrics = []
    all_miou = []
    max_iou,max_epoch = 0,0

    # for epoch_num in range(1, args.max_epochs):
    for epoch_num in range(5,6):
        pth_load_path = f'{record_save_dir}/checkpoints/epoch_{epoch_num}.pth'
        print(f'load pth from {pth_load_path}')
        model.load_parameters(pth_load_path)
        all_data_precision_p,all_data_precision_n = 0., 0.
        for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
            mask_256 = sampled_batch['mask_256'].to(device)
            gt_origin_masks = sampled_batch['mask_512']
            mask_1024 = sampled_batch['mask_1024'].to(device)
            bs_image_embedding = sampled_batch['img_embed'].to(device)
            image_pe = model.prompt_encoder.get_dense_pe()
            all_segment_logits = []

            nps = args.n_per_side
            ps,pb = model.points_for_sam, model.points_per_batch
            # nps*nps 个有效点
            point_available = np.ones((nps*nps,), dtype=int)
            single_img_precision_p,single_img_precision_n = 0., 0.
            filter_times = 0
            while np.sum(point_available) != 0:
                points_new = []
                indices, = np.nonzero(point_available)
                if len(indices) >= pb:
                    selected_idx = np.random.choice(indices, size=pb, replace=False)
                else:
                    selected_idx = indices
                points_new = ps[selected_idx]
                point_available[selected_idx] = 0

                with torch.no_grad():
                    bs_coords_torch_every = torch.as_tensor(np.array(points_new), dtype=torch.float, device=device).unsqueeze(1)
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
                    sam_pred_mask_256 = low_res_logits_bs.flatten(0, 1) > 0
                    batch_data = MaskData(
                        embed_256 = embeddings_256_bs,  # shape: (points_per_batch, 32, 256, 256)
                        mask_token_out = mask_token_out_bs,  # shape: (points_per_batch, 1, 256)
                    )

                    outputs = model.forward_batch_points(batch_data)
                    all_segment_logits.append(outputs['keep_cls_logits'].detach())
                    
                    sam_pred_mask_1024 = F.interpolate(low_res_logits_bs, (1024, 1024), mode="bilinear", align_corners=False)
                    sam_pred_mask_1024,_ = torch.max((sam_pred_mask_1024.detach() > 0).squeeze(1), dim = 0)
                    logits_1024 = F.interpolate(outputs['keep_cls_logits'], (1024, 1024), mode="bilinear", align_corners=False).detach()
                    pred_1024,_ = torch.max((F.sigmoid(logits_1024) > 0.5).squeeze(1), dim = 0)
                    
                    credible_p_mask = sam_pred_mask_1024 & pred_1024
                    credible_n_mask = sam_pred_mask_1024 & (~pred_1024)
                    filter_mask_1024 = torch.zeros_like(credible_p_mask, dtype=torch.float32).to(device)
                    filter_mask_1024[credible_p_mask] = 1
                    filter_mask_1024[credible_n_mask] = -1

                    if torch.sum(torch.abs(filter_mask_1024)) > 0:
                        # image_name = sampled_batch['meta_info']['img_name'][0]
                        # img = sampled_batch['input_image'][0].permute(1,2,0).numpy()
                        # drwa_mask(image_name, img, pred_1024, 
                        #     sam_pred_mask_1024, filter_mask_1024>0, filter_mask_1024<0, bs_coords_torch_every.squeeze(1).cpu())

                        pred_p, pred_n = torch.sum(filter_mask_1024>0), torch.sum(filter_mask_1024<0)
                        whole_TP, whole_TN = torch.sum((filter_mask_1024>0) & mask_1024), torch.sum((filter_mask_1024<0) & (~mask_1024))
                        precision_p = 1. if torch.sum(pred_p | whole_TP) == 0 else (whole_TP / (pred_p + 1e-6)).item()
                        precision_n = (whole_TN / pred_n).item()

                        single_img_precision_p += precision_p
                        single_img_precision_n += precision_n
                        filter_times += 1
                        # print(f'\npred_p:{pred_p}, pred_n:{pred_n}, whole_TP:{whole_TP}, whole_TN:{whole_TN}, precision_p:{precision_p:.4f}, precision_n:{precision_n:.4f}')

                        nonzero_inx, = np.nonzero(point_available)
                        for idx in nonzero_inx:
                            point_x,point_y = ps[idx].astype(int)
                            if int(filter_mask_1024[point_y, point_x]) != 0:
                                point_available[idx] = 0
                        # print(f'filter point num: {len(nonzero_inx) - np.sum(point_available)}')
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

            all_data_precision_p += (single_img_precision_p / filter_times)
            all_data_precision_n += (single_img_precision_n / filter_times)

        metrics = test_evaluator.evaluate(len(dataloader))
        print(f'epoch: {epoch_num} ' + str(metrics) + '\n')
        all_miou.append(metrics['mIoU'])
        if metrics['mIoU'] > max_iou:
            max_iou = metrics['mIoU']
            max_epoch = epoch_num
        
        metrics.update({
            'precision_p': f'{(all_data_precision_p / len(dataloader)):.4f}',
            'precision_n': f'{(all_data_precision_n / len(dataloader)):.4f}',
        })
        all_metrics.append(f'epoch: {epoch_num}' + str(metrics) + '\n')
    # save result file
    config_file = os.path.join(record_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_metrics)
        f.write(f'\nmax_iou: {max_iou}, max_epoch: {max_epoch}\n')
        f.write(str(all_miou))

'''
python scripts/projector/channel_val_batch.py \
    --dir_name 2024_05_14_13_45_24 \
    --n_per_side 64 \
    --points_per_batch 256 \
    --max_epochs 9 \
    --dataset_name whu
'''
