'''
可视化 给 k 个点之后的, 经 logits 蒙版处理计算后得到的 target_embedding
      与 image embedding 的相似度图
'''
import torch
from torch.nn import functional as F
import numpy as np
import os
import cv2
from utils import ResizeLongestSide
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from help_func.build_sam import sam_model_registry
import matplotlib.pyplot as plt
from utils.visualization import show_mask,show_points
from models.sam.utils.amg import build_point_grid

if __name__ == "__main__":
    test_png = [
        'datasets/WHU-Building/img_dir/train/2.png',
        'datasets/WHU-Building/img_dir/train/7.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_0.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_3.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_4.png'
    ]

    kconnect = 1     # 指在 kconnect 个连通区域上取一个点
    vis_dir_tag = 'mpoints_similarity_v2'
    
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
        save_dir = f'visual_result/{vis_dir_tag}/{image_name}/kconnect_{kconnect}'
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
            bs_image_embedding = sam.image_encoder(input_images)
            
            # 对真值图进行侵蚀，去掉小的连通域，以及大连通域的边缘地区
            # 定义侵蚀的结构元素
            kernel = np.ones((2, 2), np.uint8)
            # 对二值化图像进行侵蚀操作
            mask_64 *= 255
            eroded_image = cv2.erode(mask_64.numpy(), kernel, iterations=1)

            # 获取图中所有连通区域，并在 k 个连通区域中取 1 个前景点
            # step1: 获取连通区域
            num_labels, labels = cv2.connectedComponents(eroded_image)
            # step2: 随机选择 k 个连通区域，从它的像素坐标中随机选取 1 个前景点
            choose_k = min(kconnect, num_labels - 1)
            choosen_connect_idx = np.random.choice(range(1, num_labels), choose_k, replace=False)

            bs_points = []
            for label in choosen_connect_idx:
                region = np.argwhere(labels == label).tolist()
                # 随机获取 1 个前景点坐标, 坐标格式：(y,x)
                random_positive_idx = np.random.choice(range(len(region)), 1, replace=True)
                coordinate_1024 = []
                for idx in random_positive_idx:
                    random_coordinate = region[idx]
                    mask_64[random_coordinate[0],random_coordinate[1]] = 144
                    eroded_image[random_coordinate[0],random_coordinate[1]] = 144
                    # 将改变放大到 1024 * 1024 尺度后送入 prompt encoder, 坐标格式：(x,y)
                    coordinate_1024.append([random_coordinate[1]*16, random_coordinate[0]*16])

                coords_torch = torch.as_tensor(coordinate_1024, dtype=torch.float, device=device)
                bs_points.append(coords_torch[:, None, :])
            bs_coords_torch = torch.cat(bs_points, dim = 0)
            bs_labels_torch = torch.ones(bs_coords_torch.shape[:2], dtype=torch.int, device=device)
            points = (bs_coords_torch, bs_labels_torch)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            image_pe = sam.prompt_encoder.get_dense_pe()
            # low_res_logits.shape: (bs, 1, 256, 256)
            low_res_logits, iou_pred, embeddings_64, embeddings_256, _ = sam.mask_decoder(
                image_embeddings = bs_image_embedding,
                image_pe = image_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
            )

            feat = bs_image_embedding.squeeze(0)
            ref_feat = feat.permute(1, 2, 0)
            C, h, w = feat.shape
            test_feat = feat / feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)
            low_res_masks = F.interpolate(low_res_logits, size=ref_feat.shape[0: 2], mode='bilinear', align_corners=False)
            low_res_masks = low_res_masks.flatten(2, 3)
            masks_low_res = (low_res_masks > 0.).float()
            topk_idx = torch.topk(low_res_masks, 1)[1]
            masks_low_res.scatter_(2, topk_idx, 1.0)
            target_embedding = []
            sim = []
            for i, ref_mask in enumerate(masks_low_res.cpu()):
                ref_mask = ref_mask.squeeze().reshape(ref_feat.shape[0: 2])
                # Target feature extraction
                target_feat = ref_feat[ref_mask > 0]
                if target_feat.shape[0]>0:
                    target_embedding_ = target_feat.mean(0).unsqueeze(0)
                    target_feat = target_embedding_ / target_embedding_.norm(dim=-1, keepdim=True)
                    target_embedding_ = target_embedding_.unsqueeze(0)
                    target_embedding.append(target_embedding_)

                    sim_ = target_feat @ test_feat
                    sim_ = sim_.reshape(1, 1, h, w)
                    sim_ = F.interpolate(sim_, scale_factor=4, mode="bilinear")
                    sim_ = sam.postprocess_masks(
                                    sim_,
                                    input_size=(1024,1024),
                                    original_size=(512,512),).squeeze()
                    sim_ = sim_.cpu().numpy()
                    sim.append(sim_)

            similarity_matrix = np.array(sim).mean(0)

            plt.figure(figsize=(11,11))
            plt.imshow(image_tensor.permute(1,2,0).cpu().numpy())
            show_mask(mask_torch, plt.gca())
            show_points(bs_coords_torch.view(-1, 2).cpu(), bs_labels_torch.view(-1).cpu(), plt.gca())
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/00.png')
            plt.clf()

            fig = plt.figure(figsize=(10,20))

            ax_01 = fig.add_subplot(221)
            ax_01.imshow(mask_64)
            ax_01.set_title('gt_64')
            ax_01.axis('off')

            ax_02 = fig.add_subplot(222)
            ax_02.imshow(eroded_image)
            ax_02.set_title('gt_erode_64')
            ax_02.axis('off')

            ax_2 = fig.add_subplot(212)
            ax_2.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
            ax_2.set_title('similarity_matrix')
            ax_2.axis('off')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/group_matrix_multi.png')
            plt.clf()
