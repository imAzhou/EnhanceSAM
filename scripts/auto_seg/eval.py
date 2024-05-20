import os
import torch
import argparse
from utils import set_seed, draw_pred
from models.auto_seg import AutoSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
from utils.iou_metric import IoUMetric
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()

# base args
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

# about dataset
parser.add_argument('--server_name', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--visual_pred', action='store_true')

# about model
parser.add_argument('--n_per_side', type=int, default=32)
parser.add_argument('--points_per_batch', type=int, default=64)
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

args = parser.parse_args()


def val_one_epoch(model: AutoSegNet, val_loader):
    
    model.eval()
    all_data_precision_p,all_data_precision_n = 0., 0.
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):

        mask_1024 = sampled_batch['mask_1024'].to(device)

        bs_image_embedding = model.gene_img_embed(sampled_batch)

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
            outputs = model.forward_batch_points(bs_image_embedding, points_new)
            pred_cls_logits = outputs['cls_logits'].detach()
            all_segment_logits.append(pred_cls_logits)
            pred_sam_logits = outputs['sam_logits']

            # 过滤可信区域的点以加快推理速度
            sam_pred_mask_1024 = F.interpolate(pred_sam_logits, (1024, 1024), mode="bilinear", align_corners=False)
            sam_pred_mask_1024,_ = torch.max((sam_pred_mask_1024.detach() > 0).squeeze(1), dim = 0)
            logits_1024 = F.interpolate(pred_cls_logits, (1024, 1024), mode="bilinear", align_corners=False)
            pred_1024,_ = torch.max((F.sigmoid(logits_1024) > 0.5).squeeze(1), dim = 0)
            
            credible_p_mask = sam_pred_mask_1024 & pred_1024
            credible_n_mask = sam_pred_mask_1024 & (~pred_1024)
            filter_mask_1024 = torch.zeros_like(credible_p_mask, dtype=torch.float32).to(device)
            filter_mask_1024[credible_p_mask] = 1
            filter_mask_1024[credible_n_mask] = -1
            if torch.sum(torch.abs(filter_mask_1024)) > 0:
                # image_name = sampled_batch['meta_info']['img_name'][0]
                # img = sampled_batch['input_image'][0].permute(1,2,0).numpy()
                # batch_points = outputs['batch_points']
                # drwa_mask(image_name, img, pred_1024, 
                #     sam_pred_mask_1024, filter_mask_1024>0, filter_mask_1024<0, batch_points)

                pred_p, pred_n = torch.sum(filter_mask_1024>0), torch.sum(filter_mask_1024<0)
                whole_TP, whole_TN = torch.sum((filter_mask_1024>0) & mask_1024), torch.sum((filter_mask_1024<0) & (~mask_1024))
                precision_p = 1. if torch.sum(pred_p | whole_TP) == 0 else (whole_TP / (pred_p + 1e-6)).item()
                precision_n = (whole_TN / (pred_n + 1e-6)).item()

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
        gt_origin_masks = sampled_batch['mask_512']

        if args.visual_pred and i_batch % 50 == 0:
            pred_mask_1024 = (outputs['pred_mask_1024'].squeeze(0) > 0).detach()
            draw_pred(sampled_batch, pred_mask_1024[0], pred_save_dir)

        # gt_origin_masks[gt_origin_masks == 0] = 255
        test_evaluator.process(pred_mask, gt_origin_masks)

        all_data_precision_p += (single_img_precision_p / filter_times)
        all_data_precision_n += (single_img_precision_n / filter_times)
    
    metrics = test_evaluator.evaluate(len(val_loader))
    metrics.update({
        'precision_p': f'{(all_data_precision_p / len(val_loader)):.4f}',
        'precision_n': f'{(all_data_precision_n / len(val_loader)):.4f}',
    })
    return metrics

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = 'logs/auto_seg'
    
    # register model
    sam_ckpt = dict(
        zucc = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth',
        hz = 'checkpoints_sam/sam_vit_h_4b8939.pth'
    )
    model = AutoSegNet(
        num_classes = args.num_classes,
        n_per_side = args.n_per_side,
        points_per_batch = args.points_per_batch,
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
    test_evaluator = IoUMetric(iou_metrics=['mIoU', 'mFscore'])
    test_evaluator.dataset_meta = metainfo

    # val
    model.load_parameters(args.ckpt_path)
    metrics = val_one_epoch(model, val_dataloader)
    del metrics['ret_metrics_class']
    print(str(metrics))


'''
python scripts/auto_seg/eval.py \
    --server_name hz \
    --dataset_name inria \
    --n_per_side 64 \
    --points_per_batch 256 \
    --use_embed \
    --device cuda:1
    --use_aug \
'''
