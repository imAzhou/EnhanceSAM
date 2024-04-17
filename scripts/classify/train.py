import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.nn import BCEWithLogitsLoss
from models.classify import BinaryClassifier
from tqdm import tqdm
from datasets.whu_building_dataset import WHUBuildingDataset
from torch.utils.data import DataLoader
from help_func.tools import set_seed

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int,
                    default=12, help='total epochs for all training datasets')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
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
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # load datasets
    train_dataset = WHUBuildingDataset(
        data_root = 'datasets/WHU-Building',
        resize_size = 1024,
        mode = 'train',
        use_embed = True,
        batch_size = 16
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    val_dataset = WHUBuildingDataset(
        data_root = 'datasets/WHU-Building',
        resize_size = 1024,
        mode = 'val',
        use_embed = True,
        batch_size = 1
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    
    max_acc,max_epoch = 0,-1
    for i_episode in range(args.epochs):
        model.train()
        checkpoint_path = f"{pth_save_dir}/{i_episode}.pth"
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i_batch, sampled_batch in progress_bar:
            bs_image_embedding = sampled_batch['img_embed'].to(device)
            bs_mask_te_1024 = sampled_batch['mask_te_1024'].to(device)
            bs_labels = (torch.sum(bs_mask_te_1024, dim=(-1,-2)) > 0).int()
            outputs = model(bs_image_embedding)
            loss = criterion(outputs, bs_labels.float().view(-1, 1))  # 标签直接作为概率值
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {i_episode}/{args.epochs-1}, Loss: {loss:.4f}")
        
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for i_batch, sampled_batch in enumerate(tqdm(valloader, ncols = 70)):
                bs_image_embedding = sampled_batch['img_embed'].to(device)
                bs_mask_te_1024 = sampled_batch['mask_te_1024'].to(device)
                bs_labels = (torch.sum(bs_mask_te_1024, dim=(-1,-2)) > 0).int()
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
