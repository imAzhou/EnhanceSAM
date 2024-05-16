import os
import torch
import argparse
from utils import set_seed, get_logger, get_train_strategy, cls_loss
from models.cls_proposal import ClsProposalNet
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
from utils.iou_metric import IoUMetric

parser = argparse.ArgumentParser()

# base args
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--sam_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth')
parser.add_argument('--max_epochs', type=int, default=12, help='maximum epoch number to train and val')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')

# about dataset
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--use_aug', action='store_true')
parser.add_argument('--use_embed', action='store_true')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--train_sample_num', type=int, default=-1)

# about model
parser.add_argument('--use_module', type=str)
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

# about train
parser.add_argument('--loss_type', type=str)
parser.add_argument('--base_lr', type=float, default=0.0001)
parser.add_argument('--warmup_epoch', default=6, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--save_each_epoch', action='store_true')

args = parser.parse_args()

def train_one_epoch(model: ClsProposalNet, train_loader, optimizer):
    
    model.train()
    len_loader = len(train_loader)
    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):
        mask_1024 = sampled_batch['mask_1024'].to(device)
        mask_512 = sampled_batch['mask_512'].to(device) # shape: [bs, 512, 512]
        if args.use_embed:
            bs_input = sampled_batch['img_embed'].to(device)
        else:
            bs_input = sampled_batch['input_image'].to(device)
        
        outputs = model(bs_input, mask_1024)
        pred_logits = outputs['pred_mask_512']  # shape: [bs, num_classes, 512, 512]
        loss = cls_loss(pred_logits=pred_logits, target_masks=mask_512, args=args)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f'iteration {i_batch}/{len_loader}: loss: {loss.item():.6f}')

def val_one_epoch(model: ClsProposalNet, val_loader):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        mask_1024 = sampled_batch['mask_1024'].to(device)
        mask_512 = sampled_batch['mask_512'].to(device) # shape: [bs, 512, 512]
        bs_input_image = sampled_batch['input_image'].to(device)
        
        outputs = model(bs_input_image, mask_1024)
        # shape: [num_classes, 1024, 1024]
        pred_logits = outputs['pred_mask_512'].squeeze(0)
        pred_mask = (pred_logits>0).detach()
        mask_512[mask_512 == 0] = 255
        test_evaluator.process(pred_mask, mask_512)
    
    metrics = test_evaluator.evaluate(len(val_loader))
    return metrics

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = 'logs/cls_proposal'
    
    # register model
    model = ClsProposalNet(
                num_classes = args.num_classes,
                useModule = args.use_module,
                sam_ckpt = args.sam_ckpt
            ).to(device)
    # data loader
    train_loader,val_dataloader,metainfo = gene_loader(
        data_tag = args.dataset_name,
        use_aug = args.use_aug,
        use_embed = args.use_embed,
        train_sample_num = args.train_sample_num,
        train_bs = args.batch_size,
        val_bs = 1
    )
    # create logger
    logger,pth_save_dir = get_logger(record_save_dir, model, args)
    # get optimizer and lr_scheduler
    optimizer,lr_scheduler = get_train_strategy(model, args)
    # get evaluator
    test_evaluator = IoUMetric(iou_metrics=['mIoU'])
    test_evaluator.dataset_meta = metainfo

    # train and val in each epoch
    all_metrics,all_miou = [],[]
    max_iou,max_epoch = 0,0
    for epoch_num in tqdm(range(args.max_epochs), ncols=70):
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
        print(f'epoch: {epoch_num} ' + str(metrics) + '\n')
        if metrics['mIoU'] > max_iou:
            max_iou = metrics['mIoU']
            max_epoch = epoch_num
            save_mode_path = os.path.join(pth_save_dir, 'best_miou.pth')
            model.save_parameters(save_mode_path)

        all_miou.append(metrics['mIoU'])
        all_metrics.append(f'epoch: {epoch_num}' + str(metrics) + '\n')
    
    print(f'max_iou: {max_iou}, max_epoch: {max_epoch}')
    # save result file
    config_file = os.path.join(record_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_metrics)
        f.write(f'\nmax_iou: {max_iou}, max_epoch: {max_epoch}\n')
        f.write(str(all_miou))

'''
python scripts/cls_proposal/main.py \
    --max_epochs 24 \
    --sam_ckpt /x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth \
    --dataset_name whu \
    --use_embed \
    --batch_size 16 \
    --use_module conv \
    --loss_type bce_bdice \
    --base_lr 0.0001 \
    --train_sample_num 900 \
    --use_aug \
    --debug_mode
'''
