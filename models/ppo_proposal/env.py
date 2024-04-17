import torch
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode,Resize
from utils import ResizeLongestSide

class Env:
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 reward_thr: float,
                 use_normalized: bool=False,
                 device=None):

        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.reward_thr = reward_thr
        self.use_normalized = use_normalized
        self.patch_vertex_coords = self._split_image(image_size, patch_size)
        
    def reset(self, mask_np_64: torch.Tensor, img_embed: torch.Tensor):
        '''
        Args:
            mask_np_64: tensor, shape is (64, 64)
            img_embed: tensor, shape is (256, 64, 64)
        '''
        self.mask_np_64 = mask_np_64
        self.action_history = []
        self.positive_ratio = self._calc_ratio(mask_np_64)
        # state.shape: (c,h,w)
        state = img_embed
        if self.use_normalized:
            mean = torch.mean(state, [-1, -2], keepdim=True)
            variance = torch.var(state, [-1, -2], keepdim=True)
            # 归一化特征图
            state = (state - mean) / torch.sqrt(variance + 1e-5)  # 添加一个小的epsilon以避免除以零
        return state
    
    def reset_specify(self, img_name):
        data_root,mode = 'datasets/WHU-Building','train'
        image_mask_path = f'{data_root}/ann_dir/{mode}/{img_name}.png'
        # load image_mask
        image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
        image_mask[image_mask == 255] = 1
        transform = ResizeLongestSide(64)
        mask = transform.apply_image(image_mask.astype(np.uint8),InterpolationMode.NEAREST)
        mask = torch.as_tensor(mask)
        self.mask_np_64 = mask
        self.action_history = []
        self.positive_ratio = self._calc_ratio(mask)

        img_tensor_path = f'{data_root}/img_dir/{mode}_tensor/{img_name}.pt'
        bs_state = torch.load(img_tensor_path)
        return bs_state

    
    def close(self):
        
        self.positive_ratio = []
        self.action_history = []

    def first_step(self, action: int, state: torch.Tensor):
        '''
        Args:
            action: use action get positiion coordinate, action ∈ [0,64]
            state: the masked image embed, shape is (256, 64, 64)
        Returns:
            state: the masked image embed, shape is (256, 64, 64)
            reward: the mIoU value of each images, float type
            done: whether to terminate action in advance
        '''
        reward, right_flag, done = 0, 0, False
        gt_mask = self.mask_np_64
        
        if action == 0 :
            done = True
            # if torch.sum(gt_mask) == 0 and len(self.action_history) == 0:
            if torch.sum(gt_mask) == 0:
                reward = 1
                right_flag = 1
        else:
            # self.action_history.append(action)
            # 计算 reward
            ratio = self.positive_ratio[action-1]
            reward = ratio if ratio > self.reward_thr else 0
            # 当前 action 选中的区域内有白块
            right_flag = int(reward > self.reward_thr)
            # right_flag = int(reward>0)
            self.positive_ratio[action-1] = 0.
            x1,y1,x2,y2 = self.patch_vertex_coords[action-1]
            state[:,y1:y2,x1:x2] = 0.
            gt_mask[y1:y2,x1:x2] = 0

        return state, reward, right_flag, done
    
    def second_step(self, first_action, second_action):
        '''
        Args:
            action: use action get positiion coordinate, action ∈ [0,64]
            state: the masked image embed, shape is (256, 64, 64)
        Returns:
            state: the masked image embed, shape is (256, 64, 64)
            reward: the mIoU value of each images, float type
            done: whether to terminate action in advance
        '''
        reward = 0
        mask_np_64 = self.mask_np_64
        if second_action == 0 :
            if self.positive_ratio[first_action] <= 0.00001:
                reward = 1
        else:
            # 64 尺寸下的 patch坐标
            x1,y1,x2,y2 = self.patch_vertex_coords[first_action]
            # 64 尺寸下的 网格点坐标
            x = x1 + (second_action-1)%8
            y = y1 + (second_action-1)//8

            reward = mask_np_64[y,x].item()
            mask_np_64[y,x] = 0
        
        return reward

    
    def _split_image(self, image_size, patch_size):
        '''
        the coords return (x,y), provided that image format is (h,w)
        '''
        if image_size % patch_size != 0:
            raise ValueError("Patch size must be a divisor of image size.")
        
        step = image_size // patch_size
        patch_centers = []
        patch_vertexs = []

        for i in range(step):
            for j in range(step):
                center_x = (j + 0.5) * patch_size
                center_y = (i + 0.5) * patch_size
                patch_centers.append((int(center_x), int(center_y)))

                patch_start_y = i * self.patch_size
                patch_start_x = j * self.patch_size
                patch_end_y = patch_start_y + self.patch_size
                patch_end_x = patch_start_x + self.patch_size
                patch_vertexs.append(
                    [patch_start_x, patch_start_y, patch_end_x, patch_end_y])

        return patch_vertexs
    
    def _calc_ratio(self, state):
        step = self.image_size // self.patch_size
        positive_pixel_ratio = []

        for i in range(step):
            for j in range(step):
                patch_start_y = i * self.patch_size
                patch_start_x = j * self.patch_size
                patch_end_y = patch_start_y + self.patch_size
                patch_end_x = patch_start_x + self.patch_size
                patch = state[patch_start_y:patch_end_y, patch_start_x:patch_end_x]
            
                # 统计当前patch中非零像素的数量
                non_zero_count = torch.sum(patch).item()
                ratio = round(non_zero_count / (self.patch_size**2), 4)
                positive_pixel_ratio.append(ratio)
        return positive_pixel_ratio

    def _vis_state(self, state):
        state[state==1] = 255
        state[state==-1] = 144
        # 转换为PIL Image对象
        image = Image.fromarray(state.byte().cpu().numpy())
        # 保存图像到指定地址
        img_path = f"vis.png"
        image.save(img_path)
