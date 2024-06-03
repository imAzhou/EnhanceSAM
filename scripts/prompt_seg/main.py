import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy, cls_loss
from models.prompt_seg import PromptSegNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
from utils.iou_metric import IoUMetric

parser = argparse.ArgumentParser()

# base args
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_epochs', type=int, default=1, help='maximum epoch number to train and val')
parser.add_argument('--prompt_types', nargs='*', default=[], help='a list of prompt type,eg: box point')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')

# about dataset
parser.add_argument('--dataset_domain', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_aug', action='store_true')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--train_bs', type=int, default=16, help='training batch_size per gpu')
parser.add_argument('--val_bs', type=int, default=1, help='validation batch_size per gpu')
parser.add_argument('--train_sample_num', type=int, default=-1)
parser.add_argument('--train_load_parts', nargs='*', type=int, default=[])
parser.add_argument('--val_load_parts', nargs='*', type=int, default=[])

# about model
parser.add_argument('--semantic_module_depth', type=int)
parser.add_argument('--use_inner_feat', action='store_true')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

# about train
parser.add_argument('--loss_type', type=str)
parser.add_argument('--base_lr', type=float, default=0.0001)
parser.add_argument('--warmup_epoch', default=6, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--save_each_epoch', action='store_true')

args = parser.parse_args()

def train_one_epoch(model: PromptSegNet, train_loader, optimizer, logger):

    model.train()
    len_loader = len(train_loader)
    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        mask_512 = sampled_batch['mask_512'].to(device) # shape: [bs, 512, 512]
        mask_1024 = sampled_batch['mask_1024'] # shape: [bs, 1024, 1024]
        
        bs_image_embedding,inter_feature = model.gene_img_embed(sampled_batch)
        loss = 0.
        if 'all_boxes' in args.prompt_types:
            mask_1024_binary = sampled_batch['mask_1024_binary'] # shape: [bs, 1024, 1024]
            # all_boxes_low_logits.shape: (k_boxes, num_cls, 512, 512)
            all_boxes_low_logits, _ = model.forward_single_class(bs_image_embedding,inter_feature, mask_1024_binary)
            # pred_logits.shape: (num_cls, 512, 512)
            pred_logits,_ = torch.max(all_boxes_low_logits, dim=0)
            pred_logits = pred_logits.unsqueeze(0)  # (1, num_cls, 512, 512)
            loss = cls_loss(pred_logits=pred_logits, target_masks=mask_512, args=args)
        else:
            for cls_i in range(model.num_classes):
                mask_512_cls_i = (mask_512 == (1 if model.num_classes == 1 else cls_i)).to(torch.uint8)
                mask_1024_cls_i = (mask_1024 == (1 if model.num_classes == 1 else cls_i)).to(torch.uint8)
                # low_logits.shape: (bs, num_cls, 512, 512)
                low_logits, _ = model.forward_single_class(bs_image_embedding,inter_feature, mask_1024_cls_i)
                # cls_low_logits.shape: (bs, 1, 512, 512)
                cls_low_logits = low_logits[:, cls_i, ::].unsqueeze(1)
                loss += cls_loss(pred_logits=cls_low_logits, target_masks=mask_512_cls_i, args=args)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i_batch+1) % 20 == 0:
            logger.info(f'iteration {i_batch+1}/{len_loader}: loss: {loss.item():.6f}')

def val_one_epoch(model: PromptSegNet, val_loader, test_evaluator:IoUMetric):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        bs_mask_512 = sampled_batch['mask_512'].to(device) # shape: [bs, 512, 512]
        mask_1024 = sampled_batch['mask_1024'] # shape: [bs, 1024, 1024]
        bs_image_embedding,inter_feature = model.gene_img_embed(sampled_batch)
        if 'all_boxes' in args.prompt_types:
            mask_1024_binary = torch.zeros_like(mask_1024, dtype=torch.uint8)
            for idx in range(5):    # 1,2,3,4,5
                mask_1024_binary[mask_1024 == idx] = 1
            # all_boxes_low_logits.shape: (k_boxes, num_cls, 512, 512)
            all_boxes_low_logits, _ = model.forward_single_class(bs_image_embedding,inter_feature, mask_1024_binary)
            # pred_logits.shape: (num_cls, 512, 512)
            bs_pred_logits,_ = torch.max(all_boxes_low_logits, dim=0)
            bs_pred_logits = bs_pred_logits.unsqueeze(0)    # (1, num_cls, 512, 512)
        else:
            all_cls_logits = []
            for cls_i in range(model.num_classes):
                mask_1024_cls_i = (mask_1024 == (1 if model.num_classes == 1 else cls_i)).to(torch.uint8)
                # low_logits.shape: (bs, num_cls, 512, 512)
                low_logits, _ = model.forward_single_class(bs_image_embedding,inter_feature, mask_1024_cls_i)
                # cls_low_logits.shape: (bs, 1, 512, 512)
                cls_low_logits,_ = torch.max(low_logits, dim = 1, keepdim=True)
                all_cls_logits.append(cls_low_logits)
        
            # bs_pred_logits.shape: (bs, num_cls, 512, 512)
            bs_pred_logits = torch.cat(all_cls_logits, dim=1)
        
        if model.num_classes == 1:
            bs_pred_mask = (bs_pred_logits>0).detach()
        else:
            bs_pred_mask = bs_pred_logits.argmax(dim=1, keepdim=True).detach()
        
        for pred_mask, mask_512 in zip(bs_pred_mask, bs_mask_512):
            test_evaluator.process(pred_mask, mask_512)
    
    metrics = test_evaluator.evaluate(len(val_loader)*args.val_bs)
    return metrics

def main(logger_name):
    # data loader
    train_loader,val_dataloader,metainfo = gene_loader(
        dataset_domain = args.dataset_domain,
        data_tag = args.dataset_name,
        use_aug = args.use_aug,
        use_embed = args.use_embed,
        use_inner_feat = args.use_inner_feat,
        train_sample_num = args.train_sample_num,
        train_bs = args.train_bs,
        val_bs = args.val_bs,
        train_load_parts = args.train_load_parts,
        val_load_parts = args.val_load_parts,
    )
    # register model
    sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
    model = PromptSegNet(
                num_classes = args.num_classes,
                sm_depth = args.semantic_module_depth,
                use_inner_feat = args.use_inner_feat,
                use_embed = args.use_embed,
                sam_ckpt = sam_ckpt,
                device = device,
                prompt_types = args.prompt_types,
            ).to(device)
    # create logger
    logger,files_save_dir = get_logger(record_save_dir, model, args, logger_name)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    # get optimizer and lr_scheduler
    optimizer,lr_scheduler = get_train_strategy(model, args)
    # get evaluator
    test_evaluator = IoUMetric(iou_metrics=['mIoU','mFscore'], logger=logger)
    test_evaluator.dataset_meta = metainfo

    # train and val in each epoch
    all_metrics,all_miou = [],[]
    max_iou,max_epoch = 0,0
    for epoch_num in range(args.max_epochs):
        # train
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"epoch: {epoch_num}, learning rate: {current_lr:.6f}")
        train_one_epoch(model, train_loader, optimizer, logger)
        lr_scheduler.step()
        if args.save_each_epoch:
            save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
            model.save_parameters(save_mode_path)
            logger.info(f"save model to {save_mode_path}")

        # val
        metrics = val_one_epoch(model, val_dataloader, test_evaluator)
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
        

if __name__ == "__main__":
    
    device = torch.device(args.device)
    record_save_dir = f'logs/prompt_seg/{args.dataset_name}'
    # set_seed(args.seed)
    # main(args.loss_type)

    # for idx,base_lr in enumerate([0.005, 0.001, 0.0005]):
    #     args.base_lr = base_lr
    #     set_seed(args.seed)
    #     logger_name = f'{idx}'
    #     main(logger_name)
    
    for idx,base_lr in enumerate([0.005, 0.001, 0.0005]):
        args.base_lr = base_lr
        for loss_type in ['loss_masks','focal_bdice','bce_bdice']:
            set_seed(args.seed)
            args.loss_type = loss_type
            logger_name = f'{loss_type}_{idx}'
            main(logger_name)

    

'''
use_inner_feat
python scripts/prompt_seg/main.py \
    --dataset_domain medical \
    --max_epochs 30 \
    --dataset_name pannuke_binary \
    --prompt_types box point \
    --train_load_parts 1 2 \
    --val_load_parts 3 \
    --use_embed \
    --use_inner_feat \
    --train_bs 16 \
    --semantic_module_depth 2 \
    --loss_type loss_masks \
    --base_lr 0.005 \
    --warmup_epoch 5 \
    --gamma 0.8 \
    --train_sample_num 400 \
    --debug_mode
    --use_aug \

use_embed all_boxes
python scripts/prompt_seg/main.py \
    --dataset_domain medical \
    --max_epochs 30 \
    --prompt_types all_boxes \
    --dataset_name pannuke \
    --train_load_parts 1 2 \
    --val_load_parts 3 \
    --num_classes 6 \
    --use_embed \
    --train_bs 1 \
    --val_bs 1 \
    --semantic_module_depth 1 \
    --loss_type ce_dice \
    --base_lr 0.001 \
    --warmup_epoch 5 \
    --gamma 0.8 \
    --train_sample_num 900 \
    --device cuda:1
    --debug_mode \
'''

