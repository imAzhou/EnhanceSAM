from utils.tools import gene_max_area_box
import torch
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.visualization import show_mask,show_points,show_box

train_dataset = BuildingDataset(
    data_root = 'source/WHU-Building',
    mode = 'train',
)

trainloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    drop_last=True)

pred_save_dir = 'visual_result/center_point_box'
os.makedirs(pred_save_dir, exist_ok=True)
for i_batch, sampled_batch in enumerate(trainloader):
    mask_1024 = sampled_batch['mask_1024'][0]
    image_name = sampled_batch['meta_info']['img_name'][0]
    img = sampled_batch['input_image'][0].permute(1,2,0).numpy()

    if torch.sum(mask_1024) > 0:
        center_point,bbox = gene_max_area_box(mask_1024.numpy())
        coords_torch = torch.as_tensor(np.array([center_point]), dtype=torch.float)
        labels_torch = torch.ones(len(coords_torch))
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        show_mask(mask_1024, ax)
        show_points(coords_torch, labels_torch, ax)
        show_box(bbox, ax)
        plt.tight_layout()
        plt.savefig(f'{pred_save_dir}/{image_name}')
        plt.close()
        
