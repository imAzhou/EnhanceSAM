import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy, cls_loss
from models.auto_seg import AutoSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
import matplotlib.pyplot as plt
from utils.iou_metric import IoUMetric
from utils.visualization import show_mask,show_points
from models.sam.utils.amg import batch_iterator
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()

# base args
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_epochs', type=int, default=12, help='maximum epoch number to train and val')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')

# about dataset
parser.add_argument('--server_name', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_aug', action='store_true')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--train_sample_num', type=int, default=-1)

# about model
parser.add_argument('--n_per_side', type=int, default=32)
parser.add_argument('--points_per_batch', type=int, default=64)
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

# about train
parser.add_argument('--loss_type', type=str)
parser.add_argument('--base_lr', type=float, default=0.0001)
parser.add_argument('--warmup_epoch', default=6, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--sample_point_train', action='store_true')
parser.add_argument('--save_each_epoch', action='store_true')

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
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def train_one_epoch(model: AutoSegNet, train_loader, optimizer):
    
    model.train()
    len_loader = len(train_loader)
    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        gt_mask_256 = sampled_batch['mask_256'].to(device)
        bs_image_embedding = model.gene_img_embed(sampled_batch)

        ps,pb = model.points_for_sam, model.points_per_batch
        if args.sample_point_train:
            mask_1024 = sampled_batch['mask_1024'][0]   # mask_1024.shape: (1024,1024)
            ps_pos_neg = np.array([mask_1024[point_y,point_x] for point_x,point_y in ps.astype(int)])
            positive_num,negative_num = 128,256
            positive_coords_idx, = np.nonzero(ps_pos_neg)
            negative_coords_idx, = np.where(ps_pos_neg == 0)
            if positive_coords_idx.shape[0] > positive_num:
                random_pos_index = np.random.choice(positive_coords_idx, size=positive_num, replace=False)
            else:
                random_pos_index = positive_coords_idx
            
            if negative_coords_idx.shape[0] > negative_num:
                random_neg_index = np.random.choice(negative_coords_idx, size=negative_num, replace=False)
            else:
                random_neg_index = negative_coords_idx
            
            ps_pos = [ps[idx] for idx in random_pos_index]
            ps_neg = [ps[idx] for idx in random_neg_index]
            ps = np.array([*ps_pos, *ps_neg])

        for (batch_idx, (point_batch,)) in enumerate(batch_iterator(pb, ps)):
            outputs = model.forward_batch_points(bs_image_embedding, point_batch)
            pred_cls_logits = outputs['cls_logits']    # shape: (points_per_batch, 1, 256, 256)
            pred_sam_logits = outputs['sam_logits']
            sam_pred_mask_256 = pred_sam_logits.flatten(0, 1) > 0
            seg_gt = sam_pred_mask_256 & gt_mask_256   # shape: (points_per_batch, 256, 256)
            
            loss = cls_loss(pred_logits=pred_cls_logits, target_masks=seg_gt, args=args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f'iteration {i_batch}/{len_loader}: loss: {loss.item():.6f}')

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

        # drwa_pred(sampled_batch, pred_mask)
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
        use_aug = args.use_aug,
        use_embed = args.use_embed,
        train_sample_num = args.train_sample_num,
        train_bs = 1,
        val_bs = 1
    )
    # create logger
    logger,files_save_dir = get_logger(record_save_dir, model, args)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    # get optimizer and lr_scheduler
    optimizer,lr_scheduler = get_train_strategy(model, args)
    # get evaluator
    test_evaluator = IoUMetric(iou_metrics=['mIoU', 'mFscore'], logger=logger)
    test_evaluator.dataset_meta = metainfo

    # train and val in each epoch
    all_metrics,all_miou = [],[]
    max_iou,max_epoch = 0,0
    for epoch_num in range(args.max_epochs):
        # train
        print("epoch: ",epoch_num, "  learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, train_loader, optimizer)
        lr_scheduler.step()
        if args.save_each_epoch:
            save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
            model.save_parameters(save_mode_path)
            logger.info(f"save model to {save_mode_path}")

        # val
        metrics = val_one_epoch(model, val_dataloader)
        if args.num_classes == 1:
            mIoU = metrics['ret_metrics_class']['IoU'][1]
        else:
            mIoU = metrics['mIoU']
        
        del metrics['ret_metrics_class']
        logger.info(f'epoch: {epoch_num} ' + str(metrics) + '\n')
        if mIoU > max_iou:
            max_iou = mIoU
            max_epoch = epoch_num
            save_mode_path = os.path.join(pth_save_dir, 'best_miou.pth')
            model.save_parameters(save_mode_path)

        all_miou.append(mIoU)
        all_metrics.append(f'epoch: {epoch_num}' + str(metrics) + '\n')
    
    print(f'max_iou: {max_iou}, max_epoch: {max_epoch}')
    # save result file
    config_file = os.path.join(files_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_metrics)
        f.write(f'\nmax_iou: {max_iou}, max_epoch: {max_epoch}\n')
        f.write(str(all_miou))

'''
python scripts/auto_seg/main.py \
    --server_name hz \
    --max_epochs 6 \
    --dataset_name inria \
    --n_per_side 64 \
    --points_per_batch 256 \
    --loss_type bce_bdice \
    --base_lr 0.0001 \
    --warmup_epoch 5 \
    --use_embed \
    --sample_point_train
    --train_sample_num 400 \
    --device cuda:1
    --use_aug \
    --debug_mode
'''
