import time
import os
import torch
import argparse
import torch.optim as optim
from utils import set_seed,get_parameter_number
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from models.auto_seg import ChannelProjectorNet
from utils.losses import DiceLoss, FocalLoss, BinaryDiceLoss
from utils.logger import create_logger
from utils.loss_mask import loss_masks
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader
import numpy as np
from models.sam.utils.amg import MaskData, batch_iterator

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--n_per_side', type=int, default=32)
parser.add_argument('--points_per_batch', type=int, default=64)
parser.add_argument('--sampled_save_nums', type=int, default=100)
parser.add_argument('--base_lr', type=float, default=0.0001 )
parser.add_argument('--seed', type=int, default=1234 )
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--train_sample_num', type=int, default=-1)
# parser.add_argument('--filter_points', action='store_true')
parser.add_argument('--calc_sample_loss', action='store_true')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = 'logs/projector'
    
    # register model
    model = ChannelProjectorNet(
        n_per_side = args.n_per_side,
        points_per_batch = args.points_per_batch,
        sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
    ).to(device)
    model.train()
    
    # dataset_config = dict(
    #     whu = '/nfs/zly/datasets/WHU-Building',
    #     inria = '/nfs/zly/datasets/InriaBuildingDataset'
    # )
    dataset_config = dict(
        whu = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building',
        inria = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'
    )
    # load datasets
    train_dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        mode = 'train',
        use_embed = True,
        train_sample_num = args.train_sample_num
    )
    
    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    bce_loss_fn = BCEWithLogitsLoss()
    dice_loss_fn = BinaryDiceLoss()

    b_lr = args.base_lr
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # set record files
    save_dir_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    prefix = 'debug_' if args.debug_mode else ''
    files_save_dir = f'{record_save_dir}/{prefix}{save_dir_date}'
    os.makedirs(files_save_dir, exist_ok=True)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    os.makedirs(pth_save_dir, exist_ok=True)
    # save config file
    config_file = os.path.join(files_save_dir, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        print(f'{key}: {value}\n')
        config_items.append(f'{key}: {value}\n')
    with open(config_file, 'w') as f:
        f.writelines(config_items)
    # save log file
    logger = create_logger(f'{files_save_dir}/result.log')
    parameter_cnt = get_parameter_number(model)
    logger.info(f'网络总更新参数量：{parameter_cnt}')
    logger.info(f'网络更新参数为：')
    for name,parameters in model.named_parameters():
        if parameters.requires_grad:
            logger.info(name)

    max_iterations = (len(model.points_for_sam) // model.points_per_batch) * len(trainloader)
    iter_num = 0
    for i_batch, sampled_batch in enumerate(tqdm(trainloader, ncols=50)):
        gt_mask_256 = sampled_batch['mask_256'].to(device)
        bs_image_embedding = sampled_batch['img_embed'].to(device)
        image_pe = model.prompt_encoder.get_dense_pe()
        
        cache_batch_data = MaskData()
        ps,pb = model.points_for_sam, model.points_per_batch
        for (batch_idx, (point_batch,)) in enumerate(batch_iterator(pb, ps)):
            with torch.no_grad():
                bs_coords_torch_every = torch.as_tensor(np.array(point_batch), dtype=torch.float, device=device).unsqueeze(1)
                bs_labels_torch_every = torch.ones(bs_coords_torch_every.shape[:2], dtype=torch.int, device=device)
                points_every = (bs_coords_torch_every, bs_labels_torch_every)
                sparse_embeddings_every, dense_embeddings_every = model.prompt_encoder(
                    points=points_every,
                    boxes=None,
                    masks=None,
                )
                # low_res_logits_every.shape: (points_per_batch, 1, 256, 256)
                # iou_pred_every.shape: (points_per_batch, 1)
                # embeddings_256_every.shape: (points_per_batch, 32, 256, 256)
                # mask_token_out_bs.shape: (points_per_batch, 1, 256)
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

                cache_batch_data.cat(batch_data)
                del batch_data

            outputs = model.forward_batch_points(cache_batch_data)

            pred_logits = outputs['keep_cls_logits']    # shape: (points_per_batch, 1, 256, 256)
            seg_gt = sam_pred_mask_256 & gt_mask_256   # shape: (points_per_batch, 256, 256)

            if args.calc_sample_loss:
                bce_loss, dice_loss = loss_masks(pred_logits, seg_gt.unsqueeze(1).float())
                loss = bce_loss + dice_loss
            else:
                bce_loss = bce_loss_fn(pred_logits.squeeze(1), seg_gt[:].float())
                dice_loss = dice_loss_fn(pred_logits, seg_gt.unsqueeze(1))
                loss = (1-args.dice_param)*bce_loss + args.dice_param*dice_loss

            # bce_loss = bce_loss_fn(pred_logits.squeeze(1), seg_gt[:].float())
            # dice_loss = dice_loss_fn(pred_logits, seg_gt.unsqueeze(1))
            # loss = (1-args.dice_param)*bce_loss + args.dice_param*dice_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            shift_iter = iter_num
            lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1

            cache_batch_data = MaskData()

        logger.info(
            f'bce_loss: {bce_loss:.6f}, dice_loss: {dice_loss:.6f}, loss: {loss.item():.6f},  lr: {lr_:.6f}')
        
        if (i_batch+1) % args.sampled_save_nums == 0:
            save_mode_path = os.path.join(pth_save_dir, f'epoch_{int((i_batch+1) // args.sampled_save_nums)}.pth')
            model.save_parameters(save_mode_path)
            logger.info(f"save model to {save_mode_path}")

'''
python scripts/projector/channel_train_batch.py \
    --n_per_side 64 \
    --points_per_batch 256 \
    --sampled_save_nums 100 \
    --train_sample_num 800 \
    --dataset_name whu
    --debug_mode
'''
