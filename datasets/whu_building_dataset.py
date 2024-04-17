import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode,Resize
from utils import ResizeLongestSide

class WHUBuildingDataset(Dataset):
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
        classes=('building',),
        palette=[[180, 120, 120]])
    
    def __init__(self,
                 data_root: str,
                 resize_size:int,
                 mode:str,
                 use_embed: bool=False,
                 batch_size: int=1
        ) -> None:
        super().__init__()

        assert mode in ['train','val','test'], \
        'the mode must be in ["train","val","test"]'

        self.data_root = data_root
        self.mode = mode
        self.use_embed = use_embed
        self.resize_size = resize_size
        self.transform = ResizeLongestSide(resize_size)
        self.batch_size = batch_size

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
        sam_mask_path = f'{self.data_root}/sam_mask_dir/{self.mode}/{img_name}.pt'
        data = {
            'meta_info':dict(
                img_path = img_path, 
                image_mask_path = image_mask_path,
                img_name = img_name
            ),
            'sam_mask_index': torch.load(sam_mask_path)
        }

        if self.use_embed:
            img_tensor_path = f'{self.data_root}/img_dir/{self.mode}_tensor/{img_name}.pt'
            data['img_embed'] = torch.load(img_tensor_path)
    
        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data['origin_image_tensor'] = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        # load image_mask
        image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
        image_mask[image_mask == 255] = 1
        
        transform = ResizeLongestSide(64)
        mask = transform.apply_image(image_mask.astype(np.uint8),InterpolationMode.NEAREST)
        data['mask_np_64'] = mask

        if self.mode == 'val' or self.batch_size == 1:
            # mask size equel origin, in model test will be used
            data['mask_te_origin'] = torch.as_tensor(image_mask)

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
        image = torch.nn.functional.pad(image, pad, 'constant', 0)
        mask_torch = torch.nn.functional.pad(mask_torch, pad, 'constant', 0)

        data['mask_te_1024'] = mask_torch
        data['image_tensor'] = image

        return data
