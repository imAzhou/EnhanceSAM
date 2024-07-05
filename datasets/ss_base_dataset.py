import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from utils import ResizeLongestSide
from .augment import Pad, RandomFlip, PhotoMetricDistortion
import torchvision.transforms as T
from mmengine.structures import PixelData


class SSBaseDataset(Dataset):
    '''
    semantic segmentation dataset base config, only for multi-categories dataset, 
    labeled mask png should satisfy the needs of class id belongs to [1,num_cls]
    0 is background label(means ignore region)

    ├── data_root
       ├── img_dir
       │   ├── train
       │   │   ├── xxx{img_suffix}
       │   │   ├── yyy{img_suffix}
       │   │   ├── zzz{img_suffix}
       |   ├── train_tensor
       │   │   ├── xxx.pt
       │   │   ├── yyy.pt
       │   │   ├── zzz.pt
       │   ├── val
       │   ├── val_tensor
       ├── ann_dir
       │   ├── train
       │   │   ├── xxx{seg_map_suffix}
       │   │   ├── yyy{seg_map_suffix}
       │   │   ├── zzz{seg_map_suffix}
       │   ├── val
    '''
    METAINFO = None
    
    def __init__(self,
                 data_root: str,
                 mode:str,
                 mask_resize_sizes:list[int] = [64, 256, 512],
                 use_embed: bool=False,
                 use_aug: bool=False,
                 train_sample_num: int=-1
        ) -> None:
        super().__init__()

        assert mode in ['train','val','test'], \
        'the mode must be in ["train","val","test"]'

        self.data_root = data_root
        self.mode = mode
        self.use_embed = use_embed
        self.use_aug = use_aug
        self.mask_resize_sizes = mask_resize_sizes

        all_imgs = os.listdir(f'{data_root}/img_dir/{mode}')
        if train_sample_num > 0:
            self.images = all_imgs[:train_sample_num]
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
    
        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # data['origin_image'] = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        data['original_size'] = image.shape[:2]

        # load image_mask
        image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
        # class id belongs to [1,num_cls],0 is background label, need reduce all label value by 1.
        # 255 is ingore idx for loss calculate
        image_mask[image_mask == 0] = 255
        image_mask = image_mask - 1
        image_mask[image_mask == 254] = 255

        # aug_img: tensor,(3, 1024, 1024)      aug_gt: tensor,(1024, 1024)
        if self.use_aug:
            aug_img, aug_gt = self.gene_aug_data(image, image_mask)
        else:
            aug_img, aug_gt = self.gene_origin_data(image, image_mask)
        
        # self._showimg_and_mask(aug_img, aug_gt,img_name)

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
    
    # def _showimg_and_mask(self, aug_img, aug_gt, img_name):
    #     save_dir = 'ss_img_vis'
    #     seg_local_visualizer = SegLocalVisualizer(
    #         save_dir = save_dir,
    #         classes = self.METAINFO['classes'],
    #         palette = self.METAINFO['palette'],
    #         alpha = 0.6
    #     )
    #     data_sample = SegDataSample()
    #     data_sample.set_data({
    #         'gt_sem_seg': PixelData(**{'data': aug_gt}),
    #     })
    #     seg_local_visualizer.add_datasample(img_name, aug_img, data_sample)
