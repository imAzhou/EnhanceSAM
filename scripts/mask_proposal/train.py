import time
import os
import torch
import argparse
import torch.optim as optim
from help_func.build_sam import sam_model_registry
from help_func.tools import set_seed,get_parameter_number
from models.mask_proposal import MaskProposalNet
from utils.logger import create_logger
from tqdm import tqdm
from datasets.loveda import LoveDADataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int,
                    default=1, help='number classes of dataset')
parser.add_argument('--num_queries', type=int,
                    default=100, help='output masks of network')
# parser.add_argument('--point_num', type=int,
#                     default=1000, help='output masks of network')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--sam_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--weight_decay', type=float, default=0.001)
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = 'logs/mask_proposal'
    
    # register model
    sam = sam_model_registry['vit_h'](image_size = args.img_size).to(device)
    model = MaskProposalNet(sam,
                    num_classes = args.num_classes,
                    num_queries = args.num_queries,
                    # point_num = args.point_num,
                    sam_ckpt_path = args.sam_ckpt
                ).to(device)
    model.train()
    
    # load datasets
    train_dataset = LoveDADataset(
        data_root = '/x22201018/datasets/RemoteSensingDatasets/LoveDA',
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


    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
    else:
        b_lr = args.base_lr
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr)

    # set record files
    save_dir_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    files_save_dir = f'{record_save_dir}/{save_dir_date}'
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
            image_batch, label_batch = sampled_batch['image_tensor'].to(device), sampled_batch['gt_mask'].to(device)
            bs_image_embedding = sampled_batch['img_embed'].to(device)

            loss_dict = model.loss(bs_image_embedding, label_batch)
            loss = 0.0
            print_loss = ''
            for key,value in loss_dict.items():
                loss += value.sum()
                print_loss += f'{key}: {value.item()} , '

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
            logger.info(f'iteration {iter_num%max_iterations}/{max_iterations} : {print_loss}loss : {loss.item():.6f},  lr: {lr_:.6f}')

        save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
        model.save_parameters(save_mode_path)
        logger.info(f"save model to {save_mode_path}")

'''
python scripts/mask_proposal/train.py \
    --batch_size 16 \
    --num_classes 7 \
    --max_epochs 24 \
    --device cuda:1 \
    --base_lr 0.0001 \
    --num_queries 50 \
'''
