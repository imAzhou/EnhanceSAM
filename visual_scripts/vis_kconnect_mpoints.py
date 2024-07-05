'''
可视化 k 个连通区域分别给 m 个点之后的 logits 和 image embedding
'''
import torch
import numpy as np
import os
import cv2
from utils import ResizeLongestSide
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from help_func.build_sam import sam_model_registry
import matplotlib.pyplot as plt
from utils.visualization import show_mask,show_points

if __name__ == "__main__":
    test_png = [
        'datasets/WHU-Building/img_dir/train/2.png',
        'datasets/WHU-Building/img_dir/train/7.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_0.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_3.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_4.png'
    ]

    mpoints = 2
    
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
        save_dir = f'visual_result/kconnect_mpoints/{image_name}/mpoints_{mpoints}'
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


        with torch.no_grad():
            # input_images.shape: [bs, 3, 1024, 1024]
            input_images = sam.preprocess(image_tensor.unsqueeze(0))
            # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
            # attn_score.shape: (num_heads, patch_nums, patch_nums)
            bs_image_embedding = sam.image_encoder(input_images)
            
            # 对真值图进行侵蚀，去掉小的连通域，以及大连通域的边缘地区
            # 定义侵蚀的结构元素
            kernel = np.ones((2, 2), np.uint8)
            # 对二值化图像进行侵蚀操作
            mask_64 *= 255
            eroded_image = cv2.erode(mask_64.numpy(), kernel, iterations=1)

            # 获取图中所有连通区域，并在每个连通区域中取 mpoints 个前景点
            # step1: 获取连通区域
            num_labels, labels = cv2.connectedComponents(eroded_image)
            # step2: 从每个连通区域的像素坐标中随机选取 mpoints 个前景点
            bs_points = []
            for label in range(1, num_labels):
                region = np.argwhere(labels == label).tolist()
                # 随机获取mpoints个前景点坐标, 坐标格式：(y,x)
                random_positive_idx = np.random.choice(range(len(region)), mpoints, replace=True)
                coordinate_1024 = []
                for idx in random_positive_idx:
                    random_coordinate = region[idx]
                    mask_64[random_coordinate[0],random_coordinate[1]] = 144
                    eroded_image[random_coordinate[0],random_coordinate[1]] = 144
                    # 将改变放大到 1024 * 1024 尺度后送入 prompt encoder, 坐标格式：(x,y)
                    coordinate_1024.append([random_coordinate[1]*16, random_coordinate[0]*16])

                coords_torch = torch.as_tensor(coordinate_1024, dtype=torch.float, device=device)
                bs_points.append(coords_torch[None, :, :])
            bs_coords_torch = torch.cat(bs_points, dim = 0)
            bs_labels_torch = torch.ones(bs_coords_torch.shape[:2], dtype=torch.int, device=device)
            
            points = (bs_coords_torch, bs_labels_torch)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            image_pe = sam.prompt_encoder.get_dense_pe()
            # low_res_masks.shape: (bs, 1, 256, 256)
            low_res_masks, iou_pred, embeddings_64, embeddings_256 = sam.mask_decoder(
                image_embeddings = bs_image_embedding,
                image_pe = image_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
            )

            plt.figure(figsize=(11,11))
            plt.imshow(image_tensor.permute(1,2,0).cpu().numpy())
            show_mask(mask_torch, plt.gca())
            show_points(bs_coords_torch.view(-1, 2).cpu(), bs_labels_torch.view(-1).cpu(), plt.gca())
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/00.png')
            plt.clf()

            fig = plt.figure(figsize=(10,30))

            ax_01 = fig.add_subplot(421)
            ax_01.imshow(mask_64)
            ax_01.set_title('gt_64')
            ax_01.axis('off')

            ax_02 = fig.add_subplot(422)
            ax_02.imshow(eroded_image)
            ax_02.set_title('gt_erode_64')
            ax_02.axis('off')
            
            ax_03 = fig.add_subplot(423)
            channel_mean_encoder_64 = torch.mean(bs_image_embedding, dim=1)
            kconnect_max_encoder_64, _ = torch.max(channel_mean_encoder_64, dim=0)
            kconnect_max_encoder_64 = kconnect_max_encoder_64.cpu().numpy()
            ax_03.imshow(kconnect_max_encoder_64, cmap='hot', interpolation='nearest')
            ax_03.set_title('encoder_embed_64')
            ax_03.axis('off')

            ax_04 = fig.add_subplot(424)
            channel_mean_64 = torch.mean(embeddings_64, dim=1)
            kconnect_max_64, _ = torch.max(channel_mean_64, dim=0)
            kconnect_max_64 = kconnect_max_64.cpu().numpy()
            ax_04.imshow(kconnect_max_64, cmap='hot', interpolation='nearest')
            ax_04.set_title('decoder_embed_64')
            ax_04.axis('off')


            ax_1 = fig.add_subplot(413)
            channel_mean = torch.mean(embeddings_256, dim=1)
            kconnect_max, _ = torch.max(channel_mean, dim=0)
            kconnect_max = kconnect_max.cpu().numpy()
            ax_1.imshow(kconnect_max, cmap='hot', interpolation='nearest')
            ax_1.set_title('embed_256')
            ax_1.axis('off')

            ax_2 = fig.add_subplot(414)
            logits_256,_ = torch.max(low_res_masks.squeeze(1), dim=0)
            logits_256 = logits_256.cpu().numpy()
            ax_2.imshow(logits_256, cmap='hot', interpolation='nearest')
            ax_2.set_title('logits_256')
            ax_2.axis('off')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/group.png')
            plt.clf()
