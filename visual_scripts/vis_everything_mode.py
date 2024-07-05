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
from utils.visualization import show_mask,show_points,show_anns
from models.sam.utils.amg import build_point_grid,batch_iterator
from models.sam import SamAutomaticMaskGenerator

if __name__ == "__main__":
    test_png = [
        'datasets/WHU-Building/img_dir/train/2.png',
        'datasets/WHU-Building/img_dir/train/7.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_0.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_3.png',
        'datasets/InriaBuildingDataset/img_dir/train/austin6_0_4.png'
    ]
    
    points_per_batch = 64
    n_per_side = 32
    point_grids = build_point_grid(n_per_side)
    points_for_sam = point_grids*1024
    
    device = torch.device('cuda:0')
    image_size = 1024
    transform = ResizeLongestSide(image_size)

    # register model
    sam = sam_model_registry['vit_h'](image_size = image_size,
                                        checkpoint = 'checkpoints_sam/sam_vit_h_4b8939.pth',
                                    ).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam.eval()
    for img_path in test_png:
        mask_path = img_path.replace("img_dir", "ann_dir")
        image_name = img_path.split('/')[-1].split('.')[0]
        save_dir = f'visual_result/everything_mode/{image_name}/nperside_{n_per_side}'
        os.makedirs(save_dir,exist_ok=True)
        image = cv2.imread(img_path)
        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        auto_gene_masks = mask_generator.generate(cv2_image)

        input_image = transform.apply_image(cv2_image,InterpolationMode.BILINEAR)
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
            bs_coords_torch, bs_labels_torch = [],[]
            bs_embeddings_64, bs_embeddings_256, bs_low_res_masks = [],[],[]
            for (point_batch,) in batch_iterator(points_per_batch, points_for_sam):
                bs_coords_torch_every = torch.as_tensor(point_batch, dtype=torch.float, device=device).unsqueeze(1)
                bs_labels_torch_every = torch.ones(bs_coords_torch_every.shape[:2], dtype=torch.int, device=device)
                points_every = (bs_coords_torch_every, bs_labels_torch_every)
                sparse_embeddings_every, dense_embeddings_every = sam.prompt_encoder(
                    points=points_every,
                    boxes=None,
                    masks=None,
                )
                image_pe = sam.prompt_encoder.get_dense_pe()
                # low_res_logits_every.shape: (points_per_batch, 1, 256, 256)
                # iou_pred_every.shape: (points_per_batch, 1)
                # embeddings_256_every.shape: (points_per_batch, 32, 256, 256)
                low_res_logits, iou_pred, embeddings_64, embeddings_256 = sam.mask_decoder(
                    image_embeddings = bs_image_embedding,
                    image_pe = image_pe,
                    sparse_prompt_embeddings = sparse_embeddings_every,
                    dense_prompt_embeddings = dense_embeddings_every,
                )

                bs_coords_torch.append(bs_coords_torch_every.cpu())
                bs_labels_torch.append(bs_labels_torch_every.cpu())
                bs_embeddings_64.append(embeddings_64.cpu())
                bs_embeddings_256.append(embeddings_256.cpu())
                bs_low_res_masks.append(low_res_logits.cpu())
            bs_coords_torch = torch.cat(bs_coords_torch, dim = 0)
            bs_labels_torch = torch.cat(bs_labels_torch, dim = 0)
            bs_embeddings_64 = torch.cat(bs_embeddings_64, dim = 0)
            bs_embeddings_256 = torch.cat(bs_embeddings_256, dim = 0)
            bs_low_res_masks = torch.cat(bs_low_res_masks, dim = 0)
            

            plt.figure(figsize=(11,11))
            plt.imshow(image_tensor.permute(1,2,0).cpu().numpy())
            show_mask(mask_torch, plt.gca())
            show_points(bs_coords_torch.view(-1, 2).cpu(), bs_labels_torch.view(-1).cpu(), plt.gca(), marker_size=100)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/00_gt.png')
            plt.clf()

            plt.figure(figsize=(11,11))
            plt.imshow(cv2_image)
            show_anns(auto_gene_masks)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/00_sam_seg.png')
            plt.clf()

            fig = plt.figure(figsize=(10,20))

            ax_03 = fig.add_subplot(321)
            channel_mean_encoder_64 = torch.mean(bs_image_embedding, dim=1)
            kconnect_max_encoder_64, _ = torch.max(channel_mean_encoder_64, dim=0)
            kconnect_max_encoder_64 = kconnect_max_encoder_64.cpu().numpy()
            ax_03.imshow(kconnect_max_encoder_64, cmap='hot', interpolation='nearest')
            ax_03.set_title('encoder_embed_64')
            ax_03.axis('off')

            ax_04 = fig.add_subplot(322)
            channel_mean_64 = torch.mean(bs_embeddings_64, dim=1)
            kconnect_max_64, _ = torch.max(channel_mean_64, dim=0)
            kconnect_max_64 = kconnect_max_64.cpu().numpy()
            ax_04.imshow(kconnect_max_64, cmap='hot', interpolation='nearest')
            ax_04.set_title('decoder_embed_64')
            ax_04.axis('off')


            ax_1 = fig.add_subplot(312)
            channel_mean = torch.mean(bs_embeddings_256, dim=1)
            kconnect_max, _ = torch.max(channel_mean, dim=0)
            kconnect_max = kconnect_max.cpu().numpy()
            ax_1.imshow(kconnect_max, cmap='hot', interpolation='nearest')
            ax_1.set_title('embed_256')
            ax_1.axis('off')

            ax_2 = fig.add_subplot(313)
            logits_256,_ = torch.max(bs_low_res_masks.squeeze(1), dim=0)
            logits_256 = logits_256.cpu().numpy()
            ax_2.imshow(logits_256, cmap='hot', interpolation='nearest')
            ax_2.set_title('logits_256')
            ax_2.axis('off')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/group.png')
            plt.clf()
