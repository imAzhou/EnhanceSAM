'''
第零步：使用暴力采点的方式让 SAM 对每张图自动分割，记录每个像素点所在的 mask 索引
第一步：从 pred_logits map 中取 k=50 个候选点
第二步：计算 50 个点对应 mask 从 image embadding 中截取到的特征向量(在宽高维度上求平均)
第三步：定义 num_class = 1 个 class embedding，拼接 clas embedding 和 point feature vector
第四步：训练全连接网络做二分类判别
'''
import time
import os
import torch
import argparse
import torch.optim as optim
from help_func.build_sam import sam_model_registry
from help_func.tools import set_seed,get_parameter_number
from models.cls_proposal import ClsProposalNet,DiscriminatorNet
from tqdm import tqdm
from datasets.whu_building_dataset import WHUBuildingDataset
# from datasets.loveda import LoveDADataset
from torch.nn import BCEWithLogitsLoss
from utils.losses import FocalLoss
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
# from scripts.cls_proposal.point_discriminate_utils import sample_points_from_mask
from utils.logger import create_logger
from utils.visualization import show_mask,show_points

parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type=str)
parser.add_argument('--epoch_num', type=str)
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--logits_thr', type=float, default=0.6,
                    help='the threshold of coarse pred logits')
parser.add_argument('--sample_points', type=int, default=10,
                    help='the sampled points from coarse mask')
parser.add_argument('--max_epochs', type=int,
                    default=12, help='maximum epoch number to train')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--sam_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual', action='store_true', help='If activated, the predict mask will be saved')
args = parser.parse_args()

def train_epoch(trainloader, iter_num):
    discriminator_net.train()
    for i_batch, sampled_batch in enumerate(trainloader):
       
        # label_batch.shape: (bs, 1024, 1024)
        image_batch, label_batch = sampled_batch['image_tensor'].to(device), sampled_batch['gt_mask'].to(device)
        # if torch.all(label_batch == 0):
        #     print()
        # sam_seg_mask 的大小和原图是一样的，不一定是 1024*1024, shape: (bs, h, w)
        sam_seg_mask = sampled_batch['sam_mask_index'].to(device).unsqueeze(1)
        sam_seg_mask = F.interpolate(
            sam_seg_mask,
            (1024, 1024),
            mode="nearest"
        ).squeeze(1).long()
        bs_image_embedding = sampled_batch['img_embed'].to(device)
        bs = bs_image_embedding.shape[0]

        # 得到第一步的 pred_logits，它的尺寸是 1024*1024
        outputs = cls_proposal_net(bs_image_embedding)
        pred_logits = outputs['pred_logits']  # shape: [bs, 1, h=1024, w=1024]
        coarse_mask = (torch.sigmoid(pred_logits) > args.logits_thr).squeeze(1)

        # point_logits.shape: [bs, num_points, 1]
        point_logits, point_coords = discriminator_net(
            coarse_mask, sam_seg_mask, bs_image_embedding)
        # point_coords *= 4
        # values_at_coordinates.shape: [bs, num_points]
        values_at_coordinates = label_batch[torch.arange(bs).view(-1, 1), point_coords[:, :, 0], point_coords[:, :, 1]]

        # loss = bce_loss_fn(point_logits.squeeze(-1), values_at_coordinates[:].float())
        loss = focal_loss_fn(point_logits.squeeze(-1), values_at_coordinates[:].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.warmup and iter_num < args.warmup_period:
            lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
        else:
            if args.warmup:
                shift_iter = iter_num - args.warmup_period
                assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
            else:
                shift_iter = iter_num
            lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.95  # learning rate adjustment depends on the max iterations
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

        iter_num = iter_num + 1

        logger.info(f'iteration {iter_num%max_iterations}/{max_iterations} : loss : {loss.item():.6f},  lr: {lr_:.6f}')
    return iter_num

def val_epoch(valloader):
    discriminator_net.eval()
    total_point_nums,total_acc_point_nums = 0,0
    for i_batch, sampled_batch in enumerate(valloader):
       
        # label_batch.shape: (bs, 1024, 1024)
        image_batch, label_batch = sampled_batch['image_tensor'].to(device), sampled_batch['gt_mask'].to(device)
        # sam_seg_mask 的大小和原图是一样的，不一定是 1024*1024, shape: (bs, h, w)
        sam_seg_mask = sampled_batch['sam_mask_index'].to(device).unsqueeze(1)
        sam_seg_mask = F.interpolate(
            sam_seg_mask,
            (1024, 1024),
            mode="nearest"
        ).squeeze(1).long()
        bs_image_embedding = sampled_batch['img_embed'].to(device)
        bs = bs_image_embedding.shape[0]

        # 得到第一步的 low_res_masks，它的尺寸是 256*256
        outputs = cls_proposal_net(bs_image_embedding)
        pred_logits = outputs['pred_logits']  # shape: [bs, 1, h=256, w=246]
        coarse_mask = (torch.sigmoid(pred_logits) > args.logits_thr).squeeze(1)

        # point_logits.shape: [bs, num_points, 1]
        point_logits, point_coords = discriminator_net(
            coarse_mask, sam_seg_mask, bs_image_embedding)
        # point_coords *= 4
        # values_at_coordinates.shape: [bs, num_points]
        values_at_coordinates = label_batch[torch.arange(bs).view(-1, 1), point_coords[:, :, 0], point_coords[:, :, 1]]
        point_pred_positivity = (torch.sigmoid(point_logits) > 0.7).squeeze(-1)
        acc_count = torch.sum(point_pred_positivity == values_at_coordinates).item()

        total_acc_point_nums += acc_count
        total_point_nums += bs*args.sample_points
    return total_acc_point_nums / total_point_nums

        

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = f'logs/cls_proposal/{args.dir_name}'
    
    # register model
    sam = sam_model_registry['vit_h'](image_size = args.img_size).to(device)
    cls_proposal_net = ClsProposalNet(sam,
                    num_classes = args.num_classes,
                    sam_ckpt_path=args.sam_ckpt
                ).to(device)
    pth_load_path = f'{record_save_dir}/checkpoints/epoch_{args.epoch_num}.pth'
    cls_proposal_net.load_parameters(pth_load_path)
    cls_proposal_net.eval()
    
    discriminator_net = DiscriminatorNet(
        emb_dim = 256, sample_points = args.sample_points
        ).to(device)
    
    # load datasets
    train_dataset = WHUBuildingDataset(
        data_root = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building',
        resize_size = args.img_size,
        mode = 'train',
        use_embed = True
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    
    val_dataset = WHUBuildingDataset(
        data_root = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building',
        resize_size = args.img_size,
        mode = 'val',
        use_embed = True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    
    bce_loss_fn = BCEWithLogitsLoss()
    focal_loss_fn = FocalLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, discriminator_net.parameters()), 
        lr=args.base_lr, 
        betas=(0.9, 0.999), 
        weight_decay=args.weight_decay)
    
    # set record files
    save_dir_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    files_save_dir = f'{record_save_dir}/{save_dir_date}_point_discriminate'
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
    parameter_cnt = get_parameter_number(discriminator_net)
    logger.info(f'网络总更新参数量：{parameter_cnt}')
    logger.info(f'网络更新参数为：')
    for name,parameters in discriminator_net.named_parameters():
        if parameters.requires_grad:
            logger.info(name)
    
    # set train iterations
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)
    iterator = tqdm(range(args.max_epochs), ncols=70)
    point_max_acc,acc_max_epoch = 0, -1
    all_acc = []
    for epoch_num in iterator:
        # begin train
        iter_num = train_epoch(trainloader, iter_num)
        point_acc = val_epoch(val_dataloader)
        logger.info(f'epoch {epoch_num}/{args.max_epochs} : point_acc : {point_acc:.6f}')

        all_acc.append(point_acc)
        if point_acc > point_max_acc:
            point_max_acc = point_acc
            acc_max_epoch = epoch_num
            save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(discriminator_net.state_dict(), save_mode_path)
            logger.info(f"save model to {save_mode_path}")
    # save result file
    config_file = os.path.join(files_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.write(str(all_acc))
        f.write(f'\nmax_acc: {point_max_acc}, max_epoch: {acc_max_epoch}\n')

'''
python scripts/cls_proposal/point_discriminate.py \
    --dir_name 2024_01_22_15_18_25 \
    --epoch_num 11 \
    --batch_size 16 \
    --num_classes 1 \
    --max_epochs 1 \
    --sample_points 10 \
    --base_lr 0.0001
'''
