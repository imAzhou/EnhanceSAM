import torch
import numpy as np
import torchvision.transforms as transforms

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
    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be scaled, shape is (bs, c, h, w)
        Returns:
            Tensor: Padded image.
        """
        if np.random.rand() < 0.5:
            return transforms.RandomVerticalFlip(p=1.)(img)
        else:
            return transforms.RandomHorizontalFlip(p=1.)(img)
