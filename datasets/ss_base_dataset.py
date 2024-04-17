import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from utils import SegLocalVisualizer, SegDataSample, ResizeLongestSide
from mmengine.structures import PixelData
import mmcv
from tqdm import tqdm

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
                 resize_size:int, 
                 mode:str,
                 use_embed: bool=False
        ) -> None:
        super().__init__()

        assert mode in ['train','val','test'], \
        'the mode must be in ["train","val","test"]'

        self.data_root = data_root
        self.mode = mode
        self.use_embed = use_embed
        self.resize_size = resize_size
        self.transform = ResizeLongestSide(resize_size)

        self.images = os.listdir(f'{data_root}/img_dir/{mode}')
        self.img_masks = os.listdir(f'{data_root}/ann_dir/{mode}')
   
    def __getitem__(self, index):
        filename = self.images[index]
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
            )
        }

        if self.use_embed:
            img_tensor_path = f'{self.data_root}/img_dir/{self.mode}_tensor/{img_name}.pt'
            data['img_embed'] = torch.load(img_tensor_path)
    
        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # load image_mask
        image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
        # class id belongs to [1,num_cls],0 is background label, need reduce all label value by 1.
        # 255 is ingore idx for loss calculate
        image_mask[image_mask == 0] = 255
        image_mask = image_mask - 1
        image_mask[image_mask == 254] = 255

        if self.mode == 'val':
            data['gt_origin_mask'] = image_mask

        # resize longest
        input_image = self.transform.apply_image(image,InterpolationMode.BILINEAR)
        mask = self.transform.apply_image(image_mask.astype(np.uint8),InterpolationMode.NEAREST)

        input_image_torch = torch.as_tensor(input_image)
        mask_torch = torch.as_tensor(mask)

        data['original_size'] = image.shape[:2]
        # (H,W,C) -> (C,H,W)
        image = input_image_torch.permute(2, 0, 1).contiguous()
        data['input_size'] = tuple(image.shape[-2:])

        # to pad the last 2 dimensions of the input tensor, then use 
        # (padding_left,padding_right,padding_top,padding_bottom)
        # 255 值填充在图片右下角
        if image.shape[-1] > image.shape[-2]:   # w > h
            pad = [0, 0, 0, image.shape[-1] - image.shape[-2]]
        else:   # h > w
            pad = [0, image.shape[-2] - image.shape[-1], 0, 0]
        image = torch.nn.functional.pad(image, pad, 'constant', 255)
        mask_torch = torch.nn.functional.pad(mask_torch, pad, 'constant', 255)

        data['gt_mask'] = mask_torch
        data['image_tensor'] = image

        return data

    def visual_sample_img_mask(self, sample_num: int, save_dir: str):
        assert sample_num <= len(self.images), \
        'sample num cannot surpase total number'
        
        seg_local_visualizer = SegLocalVisualizer(
            save_dir = save_dir,
            classes = self.METAINFO['classes'],
            palette = self.METAINFO['palette'],
            alpha = 0.6
        )
        random_idxs = torch.randint(0, len(self.images), (sample_num,))
        for i in tqdm(random_idxs):
            filename = self.images[i]
            img_path = f'{self.data_root}/img_dir/{self.mode}/{filename}'
            img_name = filename.split('.')[0]
            image_mask_path = f'{self.data_root}/ann_dir/{self.mode}/{img_name}.png'
            image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
            # class id belongs to [1,num_cls],0 is background label, need reduce all label value by 1.
            # 255 is ingore idx for loss calculate
            image_mask[image_mask == 0] = 255
            image_mask = image_mask - 1
            image_mask[image_mask == 254] = 255
            mask_label = torch.as_tensor(image_mask)
            origin_image = mmcv.imread(img_path)
            data_sample = SegDataSample()
            data_sample.set_data({
                'gt_sem_seg': PixelData(**{'data': mask_label}),
            })
            seg_local_visualizer.add_datasample(img_name, origin_image, data_sample)
