import time
import os
import torch
import argparse
import torch.optim as optim
from help_func.tools import set_seed,get_parameter_number

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from models.auto_seg import ProjectorNet
from utils.losses import DiceLoss, FocalLoss, BinaryDiceLoss
from utils.logger import create_logger
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--mpoints', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.0001 )
parser.add_argument('--seed', type=int, default=1234 )

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = 'logs/projector'
    
    # register model
    model = ProjectorNet().to(device)
    model.train()
    
    dataset_config = dict(
        whu = 'datasets/WHU-Building',
        inria = 'datasets/InriaBuildingDataset'
    )
    # load datasets
    train_dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        resize_size = 1024,
        mode = 'train',
        use_embed = True
    )
    
    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    bce_loss_fn = BCEWithLogitsLoss()
    # dice_loss_fn = BinaryDiceLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.base_lr, betas=(0.9, 0.999))

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
        for i_batch, sampled_batch in enumerate(tqdm(trainloader, ncols=70)):
            image_batch, mask_64 = sampled_batch['image_tensor'].to(device), sampled_batch['mask_64'].to(device)
            
            bs_image_embedding = sampled_batch['img_embed'].to(device)
            outputs = model(bs_image_embedding, mask_64)

            pred_logits = outputs['simi_logits']
            max_simi,min_simi = outputs['max_cos_simi'],outputs['min_cos_simi']

            loss = bce_loss_fn(pred_logits.squeeze(1), mask_64[:].float())
            # dice_loss = dice_loss_fn(pred_logits, label_batch.unsqueeze(1))
            # loss = (1-args.dice_param)*bce_loss + args.dice_param*dice_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num += 1

            logger.info(
                f'loss : {loss.item():.6f}, max_simi : {max_simi:.6f}, min_simi : {min_simi:.6f}')

            if iter_num == 100:
                break

        save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
        model.save_parameters(save_mode_path)
        logger.info(f"save model to {save_mode_path}")

'''
python scripts/projector/train.py \
    --debug_mode
    --max_epochs 12 \
    --num_points 0 \
'''
