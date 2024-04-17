'''
可视化 image encoder transformer 最后一层的注意力分数
'''
import torch
import numpy as np
import os
import cv2
import random
from utils import ResizeLongestSide
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from help_func.build_sam import sam_model_registry
import matplotlib.pyplot as plt
from utils.visualization import show_mask

if __name__ == "__main__":
    test_png = [
        'datasets/WHU-Building/img_dir/train/2.png',
        'datasets/WHU-Building/img_dir/train/7.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_0.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_3.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_4.png'
    ]
    
    device = torch.device('cuda:0')
    image_size = 1024
    transform = ResizeLongestSide(image_size)

    # register model
    sam = sam_model_registry['vit_h'](image_size = image_size,
                                        checkpoint = 'checkpoints_sam/sam_vit_h_4b8939.pth',
                                    ).to(device)
    sam.eval()
    for img_path in test_png:
        mask_path = img_path.replace("img_dir", "ann_dir")
        image_name = img_path.split('/')[-1].split('.')[0]
        save_dir = f'visual_result/encoder_attn_1/{image_name}'
        os.makedirs(save_dir,exist_ok=True)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = transform.apply_image(image,InterpolationMode.BILINEAR)
        image = torch.as_tensor(input_image, device=device).permute(2, 0, 1).contiguous()
        image_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image_mask[image_mask == 255] = 1
        mask_64 = resize(torch.as_tensor(image_mask).unsqueeze(0),(64,64),InterpolationMode.NEAREST)[0]
        input_mask = transform.apply_image(image_mask.astype(np.uint8),InterpolationMode.NEAREST)
        mask_torch = torch.as_tensor(input_mask)

        if image.shape[-1] > image.shape[-2]:   # w > h
            pad = [0, 0, 0, image.shape[-1] - image.shape[-2]]
        else:   # h > w
            pad = [0, image.shape[-2] - image.shape[-1], 0, 0]
        image_tensor = torch.nn.functional.pad(image, pad, 'constant', 0)
        mask_torch = torch.nn.functional.pad(mask_torch, pad, 'constant', 0)

        plt.figure(figsize=(11,11))
        plt.imshow(image_tensor.permute(1,2,0).cpu().numpy())
        show_mask(mask_torch, plt.gca())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/00.png')
        plt.clf()

        with torch.no_grad():
            # input_images.shape: [bs, 3, 1024, 1024]
            input_images = sam.preprocess(image_tensor.unsqueeze(0))
            # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
            # attn_score.shape: (num_heads, patch_nums, patch_nums)
            bs_image_embedding, attn_score = sam.image_encoder(input_images)
            num_heads, total_patch_num, _ = attn_score.shape

            # 获取连通区域
            num_labels, labels = cv2.connectedComponents(mask_64.numpy())
            # 创建一个字典来保存每个连通区域的像素坐标
            connected_regions = {}
            for label in range(1, num_labels):
                connected_regions[label] = np.argwhere(labels == label).tolist()

            # 随机选择一个点坐标
            for label, region in connected_regions.items():
                random_point = random.choice(region)
                print("Random point in region {}: {}".format(label, random_point))
                idx = random_point[0]*64 + random_point[1]
                patch_attn = torch.mean(attn_score[:, idx, :], dim=0)
                patch_attn = patch_attn.numpy().reshape(64,64)
                
                fig = plt.figure(figsize=(12,6))
                
                mask_64_255 = mask_64 * 255
                mask_64_255[random_point[0],random_point[1]] = 144
                ax_0 = fig.add_subplot(121)
                ax_0.imshow(mask_64_255)
                ax_0.set_title('GT mask')
                
                ax_1 = fig.add_subplot(122)
                ax_1.imshow(patch_attn, cmap='hot', interpolation='nearest')
                ax_1.set_title(f'head average')

            # random_idx = torch.randint(0, total_patch_num, (choice_patch_num,))
            # for idx in random_idx:
            #     fig, axs = plt.subplots(5, 4, figsize=(15, 12), sharex=True, sharey=True)
            #     for head_idx, ax in enumerate(axs.flat):
            #         if head_idx < num_heads:
            #             patch_attn = attn_score[head_idx, idx, :]
            #             patch_attn = patch_attn.numpy().reshape(64,64)
            #             im = ax.imshow(patch_attn, cmap='hot', interpolation='nearest')
            #             ax.set_title(f'head {head_idx}')
            #         elif head_idx == 16:
            #             patch_attn = torch.mean(attn_score[:, idx, :], dim=0)
            #             patch_attn = patch_attn.numpy().reshape(64,64)
            #             im = ax.imshow(patch_attn, cmap='hot', interpolation='nearest')
            #             ax.set_title(f'head average')
            #         else:
            #             ax.axis('off')
            #         # plt.close()

                # fig.colorbar(im, ax=axs.ravel().tolist())
                plt.tight_layout()
                plt.savefig(f'{save_dir}/patch_{idx}.png')
                plt.clf()
