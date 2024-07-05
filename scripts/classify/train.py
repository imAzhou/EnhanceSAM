import os
import torch
from utils import set_seed
import torch.optim as optim
import argparse
from torch.nn import BCEWithLogitsLoss
from models.classify import BinaryClassifier
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int,
                    default=12, help='total epochs for all training datasets')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--dataset_domain', type=str)
parser.add_argument('--dataset_name', type=str, default='whu')
parser.add_argument('--train_sample_num', type=int, default=-1)
parser.add_argument('--train_load_parts', nargs='*', type=int, default=[])
parser.add_argument('--val_load_parts', nargs='*', type=int, default=[])
parser.add_argument('--train_bs', type=int, default=16, help='training batch_size per gpu')
parser.add_argument('--val_bs', type=int, default=1, help='validation batch_size per gpu')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefix is debug')

args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    if(torch.cuda.is_available()):
        torch.cuda.empty_cache()
    record_save_dir = 'logs/classify'
    pth_save_dir = f'{record_save_dir}/checkpoints'
    os.makedirs(pth_save_dir, exist_ok=True)

    model = BinaryClassifier(input_channel=256).to(device)
    criterion = BCEWithLogitsLoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # load datasets
    # data loader
    train_loader,val_dataloader,metainfo = gene_loader(
        dataset_domain = args.dataset_domain,
        data_tag = args.dataset_name,
        use_aug = False,
        use_embed = True,
        use_inner_feat = False,
        train_sample_num = args.train_sample_num,
        train_bs = args.train_bs,
        val_bs = args.val_bs,
        train_load_parts = args.train_load_parts,
        val_load_parts = args.val_load_parts,
    )
    
    max_acc,max_epoch = 0,-1
    for i_episode in range(args.epochs):
        model.train()
        checkpoint_path = f"{pth_save_dir}/{i_episode}.pth"
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i_batch, sampled_batch in progress_bar:
            bs_image_embedding = sampled_batch['img_embed'].to(device)
            bs_mask_te_1024 = sampled_batch['mask_1024'].to(device)
            
            all_pixes = 1024*1024
            foreground_pixes = torch.sum(bs_mask_te_1024, dim=(-1,-2))
            foreground_ratio = foreground_pixes / all_pixes
            bs_labels = (foreground_ratio > 0.15).int()
            outputs = model(bs_image_embedding)
            loss = criterion(outputs, bs_labels.float().view(-1, 1))  # 标签直接作为概率值
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {i_episode}/{args.epochs-1}, Loss: {loss:.4f}")
        
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols = 70)):
                bs_image_embedding = sampled_batch['img_embed'].to(device)
                bs_mask_te_1024 = sampled_batch['mask_1024'].to(device)
                
                all_pixes = 1024*1024
                foreground_pixes = torch.sum(bs_mask_te_1024, dim=(-1,-2))
                foreground_ratio = foreground_pixes / all_pixes
                bs_labels = (foreground_ratio > 0.15).int()
                outputs = model(bs_image_embedding)
                predicted = (outputs > 0.5).float()  # 使用阈值0.5进行分类
                total += bs_labels.size(0)
                correct += (predicted == bs_labels.float().view(-1, 1)).sum().item()
            acc = round(correct/total, 2)
            if acc > max_acc:
                max_acc = acc
                max_epoch = i_episode
                torch.save(model.state_dict(), checkpoint_path)
                print(f"checkpoint saved in {checkpoint_path}")
        print(f"Accuracy: {correct}/{total} = {round(correct/total, 2)}")
    print(f"max accuracy: {max_acc} in epoch {max_epoch}")

'''
python scripts/classify/train.py \
    --dataset_domain medical \
    --dataset_name pannuke_binary \
    --train_load_parts 1 2 \
    --val_load_parts 3 \

'''