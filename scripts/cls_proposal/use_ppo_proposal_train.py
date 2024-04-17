import time
import os
import torch
import argparse
import torch.optim as optim
from help_func.build_sam import sam_model_registry
from help_func.tools import set_seed,get_parameter_number
from models.cls_proposal import ClsProposalNet
from models.ppo_proposal import Env, PPOAgent
from torch.nn import BCEWithLogitsLoss
from utils.losses import BinaryDiceLoss
from utils.logger import create_logger
from tqdm import tqdm
import random
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str,
                    default='whu', help='max points select in one episode')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--num_points', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=12, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--sam_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--ppo_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = 'logs/cls_proposal_with_ppo'

    ppo_agent = PPOAgent(
        action_dim = 65,
        state_dim = 64*64,
        mode='val',
        device = device
    )
    ppo_agent.load(args.ppo_ckpt)
    env = Env(
        image_size = 64,
        patch_size = 8,
        reward_thr = 0,
        use_normalized = False,
        device = device )
    
    # register model
    sam = sam_model_registry['vit_h'](image_size = args.img_size).to(device)
    model = ClsProposalNet(sam,
                    num_classes = args.num_classes,
                    num_points = args.num_points,
                    sam_ckpt_path = args.sam_ckpt
                ).to(device)
    model.train()
    
    dataset_config = dict(
        whu = 'datasets/WHU-Building',
        inria = 'datasets/InriaBuildingDataset'
    )
    # load datasets
    train_dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
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
    
    # set loss function
    # ce_loss_fn = CrossEntropyLoss(ignore_index = 255)
    # dice_loss_fn = DiceLoss(args.num_classes, ignore_index = 255)
    # focal_loss_fn = FocalLoss(balance_param= 1-args.dice_param)

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
            image_batch, label_batch = sampled_batch['image_tensor'].to(device), sampled_batch['mask_te_1024'].to(device)
            bs_image_embedding = sampled_batch['img_embed']
            bs_input_boxes = []
            with torch.no_grad():
                for i, img_emb in enumerate(bs_image_embedding):
                    state = env.reset(sampled_batch['mask_np_64'][i], img_emb)
                    prompt_boxes = []
                    for t in range(1, args.num_points+1):
                        # select action with policy
                        action = ppo_agent.select_action(state)
                        state, reward, right_flag, done = env.first_step(action, state)
                        scale_ratio = 1024 // 64
                        if action > 0:
                            x1,y1,x2,y2 = [t*scale_ratio for t in env.patch_vertex_coords[action-1]]
                            prompt_boxes.append([x1,y1,x2,y2])
                        
                        if done:
                            # 补足 box 框的长度
                            if len(prompt_boxes) > 0:
                                while len(prompt_boxes) < args.num_points:
                                    prompt_boxes.append(random.choice(prompt_boxes))
                            else:
                                prompt_boxes = [[-1,-1,-1,-1] for _ in range(args.num_points)]
                            break

                    # clear buffer
                    ppo_agent.buffer.clear()
                    bs_input_boxes.append(prompt_boxes)
                
            bs_input_boxes = torch.tensor(bs_input_boxes, device=device)
            outputs = model.forward_with_ppo(bs_image_embedding.to(device), bs_input_boxes)

            pred_logits = outputs['pred_logits']  # shape: [bs, num_classes, 1024, 1024]
            
            # multi_label_batch = torch.repeat_interleave(label_batch.unsqueeze(1), args.num_points, dim=1)
            # label_h,label_w = multi_label_batch.shape[-2:]
            # multi_label_batch = multi_label_batch.view((-1, label_h,label_w)).contiguous()
            multi_label_batch = label_batch

            bce_loss = bce_loss_fn(pred_logits.squeeze(1), multi_label_batch[:].float())
            # loss_3 = focal_loss(low_res_logits.squeeze(1), low_res_label_batch[:].float())
            dice_loss = dice_loss_fn(pred_logits, multi_label_batch.unsqueeze(1))
            loss = (1-args.dice_param)*bce_loss + args.dice_param*dice_loss
            
            # ce_loss = ce_loss_fn(pred_logits, label_batch.long())
            # target_one_hot = dice_loss_fn._one_hot_encoder(label_batch)
            # focal_loss = focal_loss_fn(pred_logits, target_one_hot)
            # dice_loss = dice_loss_fn(pred_logits,label_batch, softmax=True)


            # loss = args.dice_param * dice_loss + focal_loss
            # loss = args.dice_param * dice_loss + (1-args.dice_param) * ce_loss

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

            # logger.info(f'iteration {iter_num%max_iterations}/{max_iterations} : ce_loss : {ce_loss.item():.6f}, lr: {lr_:.6f}')
            # logger.info(f'iteration {iter_num%max_iterations}/{max_iterations} : focal_loss : {focal_loss.item():.6f}, dice_loss : {dice_loss.item():.6f}, loss : {loss.item():.6f},  lr: {lr_:.6f}')
            logger.info(f'iteration {iter_num%max_iterations}/{max_iterations} : bce_loss : {bce_loss.item():.6f}, dice_loss : {dice_loss.item():.6f}, loss : {loss.item():.6f},  lr: {lr_:.6f}')

        save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
        model.save_parameters(save_mode_path)
        logger.info(f"save model to {save_mode_path}")

'''
python scripts/cls_proposal/use_ppo_proposal_train.py \
    --batch_size 16 \
    --max_epochs 12 \
    --num_classes 1 \
    --num_points 3 \
    --dice_param 0.8 \
    --dataset_name inria \
    --base_lr 0.0001 \
    --ppo_ckpt logs/ppo_proposal/2024_03_20_10_51_02/checkpoints/34.pth
    --device cuda:1 \
    --weight_decay 0.01
'''
