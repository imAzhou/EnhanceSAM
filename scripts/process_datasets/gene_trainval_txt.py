import torch
from tqdm import tqdm
import cv2
import os

root_image_path = 'datasets/WHU-Building/img_dir'
root_mask_path = 'datasets/WHU-Building/ann_dir'
modes = ['train','val']

for mode in modes:
    total_line = 0
    img_masks = os.listdir(f'{root_mask_path}/{mode}')
    text_save = f'{root_image_path}/{mode}.txt'
    with open(text_save, 'w', encoding='utf-8') as file:
        for mask_filename in tqdm(img_masks):
            image_mask_path = f'{root_mask_path}/{mode}/{mask_filename}'
            image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
            image_mask[image_mask == 255] = 1
            image_mask = torch.as_tensor(image_mask)
            h,w = image_mask.shape[-2:]
            positive_ratio = torch.sum(image_mask) / (h*w)
            if positive_ratio > 0.05:
                file.write(f'{mask_filename}\n')
                total_line += 1
        print(f'total_line: {total_line}')