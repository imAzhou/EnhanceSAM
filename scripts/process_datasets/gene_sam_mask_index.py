import os
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from help_func.build_sam import sam_model_registry
from models.sam import SamAutomaticMaskGenerator
import numpy as np

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

data_root = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building'
save_dir = f'{data_root}/sam_mask_dir'
img_size = 1024
batch_size = 1
sam_checkpoint = "checkpoints_sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, image_size = img_size)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
all_modes = ['train', 'val']

# generate train datasets sam mask
for mode in all_modes:
    all_imgs = os.listdir(f'{data_root}/img_dir/{mode}')
    dir_name = f'{save_dir}/{mode}'
    os.makedirs(dir_name, exist_ok=True)
    for img_name in tqdm(all_imgs, ncols=100):
        filename = img_name.split('.')[0]
        # image.shape: (h,w,c)
        image = cv2.imread(f'{data_root}/img_dir/{mode}/{img_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask_index.shape: (h,w)
        
        mask_indexes = np.zeros((image.shape[:2]))
        masks = mask_generator.generate(image)
        for mask_idx, mask_info in enumerate(masks):
            seg_mask = mask_info['segmentation']
            mask_indexes[seg_mask] = mask_idx + 1
        
        mask_indexes_tensor = torch.from_numpy(mask_indexes)
        torch.save(mask_indexes_tensor, f'{dir_name}/{filename}.pt')

        # plt.figure(figsize=(20,20))
        # plt.imshow(image)
        # show_anns(masks)
        # plt.axis('off')
        # plt.savefig(img_name)
