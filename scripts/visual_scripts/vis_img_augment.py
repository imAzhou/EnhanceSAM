import torch
from datasets.augment import Pad, RandomFlip
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.visualization import show_mask


# 加载原始图像
image_path = "datasets/WHU-Building/img_dir/train/10.png"
image_mask_path = "datasets/WHU-Building/ann_dir/train/10.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.as_tensor(image).permute(2, 0, 1).contiguous()
image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
image_mask[image_mask == 255] = 1.
image_mask = torch.as_tensor(image_mask).unsqueeze(0)
# image_mask_c3 = torch.repeat_interleave(image_mask, 3, dim=0)
concat_pair = torch.cat([image, image_mask], dim=0)

# 定义图像预处理操作
common_transform = T.Compose([
    T.Resize((512, 512)),  # 缩放到指定大小
    Pad()
])

origin_pair = common_transform(concat_pair).unsqueeze(0)

flip_img = RandomFlip()(origin_pair)
rotate_img = T.RandomAffine(degrees=45)(origin_pair)
scale_img = T.RandomResizedCrop((512, 512), scale=(0.6, 1.4))(origin_pair)

cat_imgs = torch.cat([origin_pair, flip_img, rotate_img, scale_img], dim=0)

grid_image = make_grid(cat_imgs, nrow=2, padding=0)


# 显示网格图像
plt.figure(figsize=(11,11))
img = grid_image[:3].permute(1, 2, 0)
plt.imshow(img)
img_mask = grid_image[3:]
show_mask(img_mask, plt.gca())

plt.axis('off')
plt.tight_layout()
plt.savefig('img_and_mask.png')
