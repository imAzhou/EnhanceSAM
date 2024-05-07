import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from mmseg.datasets.transforms import PhotoMetricDistortion as PMD

class Pad(torch.nn.Module):
    def __init__(self, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be scaled, shape is (bs, c, h, w)
        Returns:
            Tensor: Padded image.
        """
        h,w = img.shape[-2:]
        if w > h:   # w > h
            pad = [0, 0, 0, w - h]
        else:   # h > w
            pad = [0, h - w, 0, 0]
        img = torch.nn.functional.pad(img, pad, 'constant', self.pad_value)
        return img


class RandomFlip(torch.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be scaled, shape is (bs, c, h, w)
        Returns:
            Tensor: Random flipped image.
        """
        if np.random.rand() < 0.5:
            return transforms.RandomVerticalFlip(p=self.prob)(img)
        else:
            return transforms.RandomHorizontalFlip(p=self.prob)(img)

class PhotoMetricDistortion(torch.nn.Module):
    '''
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    '''
    def __init__(self, *args):
        super().__init__()
        self.pmd = PMD(*args)

    def forward(self, img):
        """
        Args:
            img (ndarry): Image to be scaled, shape is (h,w,c), c is rgb format
        Returns:
            img (ndarry): shape is (h,w,c), c is rgb format
        """
        bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        dict_input = dict(img = bgr_image)
        dict_output = self.pmd(dict_input)
        bgr_img_np = dict_output['img']
        rgb_img_np = cv2.cvtColor(bgr_img_np, cv2.COLOR_BGR2RGB)
        return rgb_img_np

        