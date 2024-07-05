'''
可视化 某个连通区域给 m 个点之后的 logits 蒙版处理后的 image embedding，
      与 everything mode 4098 个 image emedding 之间的余弦相似度
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
from models.sam.utils.amg import build_point_grid

if __name__ == "__main__":
    test_png = [
        'datasets/WHU-Building/img_dir/train/2.png',
        'datasets/WHU-Building/img_dir/train/7.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_0.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_3.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_4.png'
    ]

    mpoints = 2
    vis_dir_tag = 'mpoints_similarity'
    
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
        save_dir = f'visual_result/{vis_dir_tag}/{image_name}/mpoints_{mpoints}'
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
            # step2: 随机选择一个连通区域，从它的像素坐标中随机选取 mpoints 个前景点
            choosen_connect_idx = np.random.choice(range(1, num_labels), 1)

            region = np.argwhere(labels == choosen_connect_idx).tolist()
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
            bs_coords_torch = coords_torch[None, :, :]
            bs_labels_torch = torch.ones(bs_coords_torch.shape[:2], dtype=torch.int, device=device)
            
            points = (bs_coords_torch, bs_labels_torch)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            image_pe = sam.prompt_encoder.get_dense_pe()
            # low_res_logits.shape: (bs, 1, 256, 256)
            low_res_logits, iou_pred, embeddings_64, embeddings_256 = sam.mask_decoder(
                image_embeddings = bs_image_embedding,
                image_pe = image_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
            )
            # (mpoints, 32, 256, 256) -> (32,)
            logits = torch.repeat_interleave(low_res_logits, embeddings_256.shape[1], dim=1)
            # embeddings_256[logits<0] = 0
            embeddings_256 += logits
            mpoints_vector = torch.mean(embeddings_256, dim=(2,3)).mean(dim=0, keepdim=True)

            n_per_side = 32
            point_grids = build_point_grid( n_per_side = n_per_side )
            # [[x,y],[x,y], ... ,[x,y]]
            points_for_sam = point_grids*image_size
            coords_torch_every = torch.as_tensor(points_for_sam, dtype=torch.float, device=device)
            bs_coords_torch_every = coords_torch_every[:, None, :]
            bs_labels_torch_every = torch.ones(bs_coords_torch_every.shape[:2], dtype=torch.int, device=device)
            points = (bs_coords_torch_every, bs_labels_torch_every)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            image_pe = sam.prompt_encoder.get_dense_pe()
            # low_res_logits_every.shape: (bs, 1, 256, 256)
            low_res_logits_every, iou_pred_every, embeddings_64_every, embeddings_256_every = sam.mask_decoder(
                image_embeddings = bs_image_embedding,
                image_pe = image_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
            )
            logits_every = torch.repeat_interleave(low_res_logits_every, embeddings_256_every.shape[1], dim=1)
            # embeddings_256_every[logits_every<0] = 0
            embeddings_256_every += logits_every
            every_vector = torch.mean(embeddings_256_every, dim=(2,3))

            cosine_similarity = torch.matmul(
                mpoints_vector, every_vector.t()) / (torch.norm(mpoints_vector) * torch.norm(every_vector))
            similarity_matrix = cosine_similarity.squeeze(0).view(n_per_side,n_per_side)

            plt.figure(figsize=(11,11))
            plt.imshow(image_tensor.permute(1,2,0).cpu().numpy())
            show_mask(mask_torch, plt.gca())
            show_points(bs_coords_torch.view(-1, 2).cpu(), bs_labels_torch.view(-1).cpu(), plt.gca())
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/00.png')
            plt.clf()

            fig = plt.figure(figsize=(10,20))

            ax_01 = fig.add_subplot(331)
            ax_01.imshow(mask_64)
            ax_01.set_title('gt_64')
            ax_01.axis('off')

            ax_02 = fig.add_subplot(332)
            ax_02.imshow(eroded_image)
            ax_02.set_title('gt_erode_64')
            ax_02.axis('off')

            ax_03 = fig.add_subplot(333)
            ax_03.imshow(similarity_matrix.cpu().numpy())
            ax_03.set_title('similarity_matrix')
            ax_03.axis('off')

            ax_2 = fig.add_subplot(312)
            logits_256,_ = torch.max(low_res_logits.squeeze(1), dim=0)
            logits_256 = logits_256.cpu().numpy()
            ax_2.imshow(logits_256, cmap='hot', interpolation='nearest')
            ax_2.set_title('logits_256')
            ax_2.axis('off')

            ax_3 = fig.add_subplot(313)
            logits_256_every,_ = torch.max(low_res_logits_every.squeeze(1), dim=0)
            logits_256_every = logits_256_every.cpu().numpy()
            ax_3.imshow(logits_256_every, cmap='hot', interpolation='nearest')
            ax_3.set_title('logits_256_every')
            ax_3.axis('off')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/group_add.png')
            plt.clf()
