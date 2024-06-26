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


class BuildingDataset(Dataset):
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
        classes=('background','building',),
        palette=[[255, 255, 255],[180, 120, 120]])
    
    def __init__(self,
                 data_root: str,
                 mode:str,
                 mask_resize_sizes:list[int] = [64, 256, 512],
                 use_embed: bool=False,
                 use_inner_feat: bool=False,
                 use_aug: bool=False,
                 train_sample_num: int=-1
        ) -> None:
        super().__init__()

        assert mode in ['train','val','test'], \
        'the mode must be in ["train","val","test"]'

        self.data_root = data_root
        self.mode = mode
        self.use_embed = use_embed
        self.use_inner_feat = use_inner_feat
        self.use_aug = use_aug
        self.mask_resize_sizes = mask_resize_sizes

        all_imgs = os.listdir(f'{data_root}/img_dir/{mode}')
        if train_sample_num > 0:
            if train_sample_num > len(all_imgs):
                raise ValueError("sample nums cannot surpass total image nums!")
            self.images = random.sample(all_imgs, train_sample_num)
        else:
            self.images = all_imgs
        
    def __getitem__(self, index):
        filename = self.images[index].strip()
        img_name = filename.split('.')[0]
        data = self.process_img_mask(img_name)

        return data

    def __len__(self):
        return len(self.images)
    
    def process_img_mask(self, img_name):
        img_path = f'{self.data_root}/img_dir/{self.mode}/{img_name}.png'
        image_mask_path = f'{self.data_root}/ann_dir/{self.mode}/{img_name}.png'
        data = {
            'meta_info':dict(
                img_path = img_path, 
                image_mask_path = image_mask_path,
                img_name = img_name
            ),
        }

        if self.use_embed:
            img_tensor_path = f'{self.data_root}/img_dir/{self.mode}_tensor/{img_name}.pt'
            data['img_embed'] = torch.load(img_tensor_path)
            if self.use_inner_feat:
                img_inner_tensor_path = f'{self.data_root}/img_dir/{self.mode}_tensor/{img_name}_inner_0.pt'
                data['img_embed_inner'] = torch.load(img_inner_tensor_path)
    
        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # data['origin_image'] = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        data['original_size'] = image.shape[:2]

        # load image_mask
        image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
        image_mask[image_mask == 255] = 1

        # aug_img: tensor,(3, 1024, 1024)      aug_gt: tensor,(1024, 1024)
        if self.use_aug:
            aug_img, aug_gt = self.gene_aug_data(image, image_mask)
        else:
            aug_img, aug_gt = self.gene_origin_data(image, image_mask)
        
        # if torch.sum(aug_gt) > 0:
        #     self._showimg_and_mask(aug_img, aug_gt)

        for size in self.mask_resize_sizes:
            transform = ResizeLongestSide(size)
            mask = transform.apply_image(aug_gt.numpy().astype(np.uint8),InterpolationMode.NEAREST)
            data[f'mask_{size}'] = mask       
        
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
            T.Resize((1024, 1024)),  # 缩放到指定大小
            Pad(),
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
        
        aug_img = Pad()(input_image)
        aug_gt = Pad()(torch.as_tensor(mask))

        return aug_img, aug_gt
    
    def _showimg_and_mask(self, aug_img, aug_gt):
        import matplotlib.pyplot as plt
        from utils.visualization import show_mask
        
        plt.figure(figsize=(11,11))
        plt.imshow(aug_img.permute(1, 2, 0))
        show_mask(aug_gt, plt.gca())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('img_and_mask_origin.png')
