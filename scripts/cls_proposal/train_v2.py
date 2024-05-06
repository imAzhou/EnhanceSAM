import time
import os
import torch
import argparse
import torch.optim as optim
from utils import (
    set_seed,get_parameter_number,
    BinaryDiceLoss
)
from models.cls_proposal_v2 import ClsProposalNet
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from utils.loss_mask import loss_masks
from utils.logger import create_logger
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_module', type=str)
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--num_points', nargs='+', type=int)
parser.add_argument('--max_epochs', type=int,
                    default=12, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--train_sample_num', type=int, default=-1)
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--use_aug', action='store_true')
parser.add_argument('--calc_sample_loss', action='store_true')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = 'logs/cls_proposal_v2'
    
    # register model
    model = ClsProposalNet(
                num_classes = args.num_classes,
                num_points = args.num_points,
                useModule = args.use_module
            ).to(device)
    model.train()
    
    dataset_config = dict(
        whu = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building',
        inria = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'
    )
    # load datasets
    train_dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        mode = 'train',
        # use_embed = True,
        use_aug = args.use_aug,
        train_sample_num = args.train_sample_num
    )
    
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    bce_loss_fn = BCEWithLogitsLoss()
    dice_loss_fn = BinaryDiceLoss()

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
    else:
        b_lr = args.base_lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr)

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
     
    # set train iterations
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)
    iterator = tqdm(range(args.max_epochs), ncols=70)

    # begin train
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            mask_1024 = sampled_batch['mask_1024'].to(device)
            mask_512 = sampled_batch['mask_512'].to(device)
            bs_input_image = sampled_batch['input_image'].to(device)
            
            outputs = model(bs_input_image, mask_1024)
            pred_logits = outputs['pred_mask_512']  # shape: [bs, num_classes, 512, 512]
            
            if args.calc_sample_loss:
                bce_loss, dice_loss = loss_masks(pred_logits, mask_512.unsqueeze(1).float())
                loss = bce_loss + dice_loss
            else:
                bce_loss = bce_loss_fn(pred_logits.squeeze(1), mask_512[:].float())
                dice_loss = dice_loss_fn(pred_logits, mask_512.unsqueeze(1))
                loss = (1-args.dice_param)*bce_loss + args.dice_param*dice_loss

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

            logger.info(f'iteration {iter_num%max_iterations}/{max_iterations} : bce_loss : {bce_loss.item():.6f}, dice_loss : {dice_loss.item():.6f}, loss : {loss.item():.6f},  lr: {lr_:.6f}')

        save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
        model.save_parameters(save_mode_path)
        logger.info(f"save model to {save_mode_path}")

'''
python scripts/cls_proposal/train_v2.py \
    --max_epochs 24 \
    --batch_size 16 \
    --num_points 1 0 \
    --base_lr 0.0001 \
    --use_module conv \
    --dataset_name whu \
    --calc_sample_loss \
    --use_aug \
    --train_sample_num 400
    --debug_mode
'''
