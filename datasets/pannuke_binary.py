import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from utils import ResizeLongestSide
from .augment import Pad, RandomFlip, PhotoMetricDistortion
import torchvision.transforms as T
import json


class PannukeBinaryDataset(Dataset):
    '''
    memo: label is gt binary mask, foreground pixel value is 255, backround is 0
    when load to trian, gt mask foreground pixel value is 1, backround is 0
    structure:
    ├── data_root(/x22201018/datasets/RemoteSensingDatasets/WHU-Building)
       ├── train
       │   ├── tensor
       │   │   ├── 1.pt (c=256,h=64,w=64)
       │   ├── label
       │   │   ├── 1.png
       ├── val
       ├── test
    '''
    METAINFO = dict(
        classes=('background','cell',),
        palette=[[255, 255, 255],[180, 120, 120]])
    
    def __init__(self,
                 data_root: str,
                 load_parts:list[int],
                 mask_resize_sizes:list[int] = [64, 256, 512],
                 use_embed: bool=False,
                 use_inner_feat: bool=False,
                 use_aug: bool=False,
                 train_sample_num: int=-1
        ) -> None:
        super().__init__()

        self.data_root = data_root
        self.use_embed = use_embed
        self.use_inner_feat = use_inner_feat
        self.use_aug = use_aug
        self.mask_resize_sizes = mask_resize_sizes

        all_imgs = []
        for i in load_parts:
            part_imgs = os.listdir(f'{data_root}/Part{i}/img_dir')
            all_imgs.extend(part_imgs)
        
        if train_sample_num > 0:
            if train_sample_num > len(all_imgs):
                raise ValueError("sample nums cannot surpass total image nums!")
            self.images = random.sample(all_imgs, train_sample_num)
        else:
            self.images = all_imgs
        
    def __getitem__(self, index):
        filename = self.images[index].strip()
        img_name = filename.split('.')[0]
        part_dir = filename.split('_')[0]
        data = self.process_img_mask(img_name, part_dir)

        return data

    def __len__(self):
        return len(self.images)
    
    def process_img_mask(self, img_name, part_dir):
        img_path = f'{self.data_root}/{part_dir}/img_dir/{img_name}.png'
        image_mask_path = f'{self.data_root}/{part_dir}/ann_dir/{img_name}.png'
        data = {
            'meta_info':dict(
                img_path = img_path, 
                image_mask_path = image_mask_path,
                img_name = img_name
            ),
        }

        image_boxes_path = f'{self.data_root}/{part_dir}/ann_boxes_dir/{img_name}.json'
        with open(image_boxes_path,'r',encoding='utf-8') as f :
            data['gt_boxes'] = json.load(f)
            data['coord_ratio'] = 1024 // 256

        if self.use_embed:
            img_tensor_path = f'{self.data_root}/{part_dir}/img_tensor/{img_name}.pt'
            data['img_embed'] = torch.load(img_tensor_path)
            if self.use_inner_feat:
                img_inner_tensor_path = f'{self.data_root}/{part_dir}/img_tensor/{img_name}_inner_0.pt'
                data['img_embed_inner'] = torch.load(img_inner_tensor_path)
    
        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # data['origin_image'] = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        data['original_size'] = image.shape[:2]

        # load image_mask
        image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
        image_mask_binary = np.zeros_like(image_mask).astype(np.uint8)
        for idx in range(1,6):    # 1,2,3,4,5
            image_mask_binary[image_mask == idx] = 1
        image_mask = image_mask_binary

        # aug_img: tensor,(3, 1024, 1024)      aug_gt: tensor,(1024, 1024)
        aug_img, aug_gt, input_size = self.gene_origin_data(image, image_mask)
        data['input_size'] = input_size
        if self.use_aug:
            aug_img, aug_gt = self.gene_aug_data(aug_img, aug_gt)
        
        # if torch.sum(aug_gt) > 0:
        #   self._showimg_and_mask(aug_img, aug_gt,img_name)

        for size in self.mask_resize_sizes:
            transform = ResizeLongestSide(size)
            mask = transform.apply_image(aug_gt.numpy().astype(np.uint8),InterpolationMode.NEAREST)
            data[f'mask_{size}'] = mask
            data[f'mask_{size}_binary'] = np.zeros_like(mask).astype(np.uint8)
            
        data['input_image'] = aug_img
        data['mask_1024'] = aug_gt

        return data
    
    def gene_aug_data(self, image: np.ndarray, image_mask: np.ndarray):
        '''
        Args:
            - image: ndarry, shape is (h,w,c), c is rgb format
            - image_mask: ndarray, shape is (h,w)
        Return:
            - aug_img: tensor, shape is (c, 1024, 1024)
            - aug_gt: tensor, shape is (1024, 1024)
        '''

        pmd_img = PhotoMetricDistortion()(image)
        image = torch.as_tensor(pmd_img).permute(2, 0, 1).contiguous()
        image_mask = torch.as_tensor(image_mask).unsqueeze(0)
        
        # 在通道维度 concat image 和 mask
        concat_pair = torch.cat([image, image_mask], dim=0).unsqueeze(0)

        common_transform = T.Compose([
            RandomFlip(prob=0.5),
            T.RandomAffine(degrees=20),
            T.RandomResizedCrop((1024, 1024), scale=(0.6, 1.4))
        ])
        origin_pair = common_transform(concat_pair).squeeze(0)
        aug_img,aug_gt = origin_pair[:3],origin_pair[3:].squeeze(0)

        return aug_img,aug_gt

    def gene_origin_data(self, image: np.ndarray, image_mask: np.ndarray):
        '''
        Args:
            - image: ndarry, shape is (h,w,c)
            - image_mask: ndarray, shape is (h,w)
        Return:
            - aug_img: tensor, shape is (c, 1024, 1024)
            - aug_gt: tensor, shape is (1024, 1024)
        '''
        transform = ResizeLongestSide(1024)
        input_image = transform.apply_image(image, InterpolationMode.BILINEAR)
        input_image = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()
        mask = transform.apply_image(image_mask.astype(np.uint8),InterpolationMode.NEAREST)

        # (h,w)
        input_size = (input_image.shape[1], input_image.shape[2])
        
        aug_img = Pad()(input_image)
        aug_gt = Pad()(torch.as_tensor(mask))

        return aug_img, aug_gt, input_size
