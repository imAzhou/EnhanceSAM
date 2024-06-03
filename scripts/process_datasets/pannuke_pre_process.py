import os
import time
import torch
from models.sam.build_sam import sam_model_registry
import cv2
import json
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from utils import show_multi_mask
from tqdm import tqdm
from datasets.gene_dataloader import gene_loader
from utils import set_seed

# random.seed(2)
root = '/x22201018/datasets/MedicalDatasets/PanNuke'
parts = ['Part1', 'Part2', 'Part3']

colors = {
    0: (255, 0, 0),   # 癌变细胞 Neoplastic cells - 红色
    1: (0, 255, 0),   # 炎症细胞 Inflammatory - 绿色
    2: (0, 0, 255),   # 结缔组织/软组织细胞 Connective/Soft tissue cells - 蓝色
    3: (255, 255, 0),  # 死亡细胞 Dead Cells - 黄色
    4: (255, 128, 0),  # 上皮细胞 Epithelial - 橙色
    # 5: (0, 0, 0)      # 背景 Background - 黑色
}
palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 128, 0], [255, 255, 255]]

def gene_mask_png():
    for idx, part in enumerate(parts):
        masks_dict_path = os.path.join(root, part, 'Masks', 'masks.json')
        image_path = os.path.join(root, part, 'Images', 'images.npy')
        mask_path = os.path.join(root, part, 'Masks', 'masks.npy')
        images = np.load(image_path)    # shape: (total_img_nums, h, w, c)
        masks = np.load(mask_path)      # shape: (total_img_nums, h, w, cls_num)

        print(f'images:{images.shape}, masks:{masks.shape}')
        
        ann_dir = f'{root}/{part}/ann_dir'
        os.makedirs(ann_dir, exist_ok=True)
        img_ann_dir = f'{root}/{part}/Masks/img_ann_dir'
        os.makedirs(img_ann_dir, exist_ok=True)
        img_dir = f'{root}/{part}/img_dir'

        with open(masks_dict_path,'r',encoding='utf-8') as f :
            masks_dict = json.load(f)
            for filename,mask_idx in tqdm(masks_dict.items(), ncols=70):
                img = cv2.imread(f'{img_dir}/{filename}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask_png = np.zeros(img.shape[:2])  # 创建一个全零的颜色掩码
                mask_ann = masks[mask_idx]
                for i in range(len(palette)):
                    mask_png[mask_ann[...,i] > 0] = i+1
                
                cv2.imwrite(f'{ann_dir}/{filename}', mask_png)

                # fig = plt.figure(figsize=(6,6))
                # ax = fig.add_subplot(111)
                # ax.imshow(img)
                # vis_mask_png = copy.deepcopy(mask_png)
                # vis_mask_png -= 1
                # vis_mask_png[vis_mask_png==-1] = 255
                # show_multi_mask(vis_mask_png, ax, palette)
                # ax.set_title('gt mask')
                # plt.tight_layout()
                # plt.savefig(f'{img_ann_dir}/{filename}')
                # plt.close()

def rename_file():
    for idx, part in enumerate(parts):
        img_dir = f'{root}/{part}/img_dir'
        ann_dir = f'{root}/{part}/ann_dir'
        all_img_filename = os.listdir(img_dir)
        for filename in tqdm(all_img_filename,ncols=70):
            old_img_file = os.path.join(img_dir, filename)
            old_ann_file = os.path.join(ann_dir, filename)
            new_filename = f'{part}_{filename}'
            new_img_file = os.path.join(img_dir, new_filename)
            new_ann_file = os.path.join(ann_dir, new_filename)
            # 重命名文件
            os.rename(old_img_file, new_img_file)
            os.rename(old_ann_file, new_ann_file)

def gene_sam_tensor():
    set_seed(1234)
    device = torch.device('cuda:0')
    image_size = 1024

    train_loader,val_dataloader,metainfo = gene_loader(
        dataset_domain = 'medical',
        data_tag = 'pannuke',
        train_load_parts = [1,2],
        val_load_parts = [3],
        use_aug = False,
        use_embed = False,
        train_sample_num = -1,
        train_bs = 1,
        val_bs =1,
    )

    # register model
    sam = sam_model_registry['vit_h'](image_size = image_size,
                                        # checkpoint = 'checkpoints_sam/sam_vit_h_4b8939.pth',
                                        checkpoint = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth',
                                    ).to(device)
    sam.eval()

    for dataloader in [train_loader, val_dataloader]:
        start = time.time()
        for batch_idx, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
            image_batch = sampled_batch['input_image'].to(device)
            img_path,img_name = sampled_batch['meta_info']['img_path'],sampled_batch['meta_info']['img_name']
            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                input_images = sam.preprocess(image_batch)
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                bs_image_embedding,inter_feature  = sam.image_encoder(input_images, need_inter=True)
            for img_emb_tensor,inner_emb_tensor, save_name in zip(bs_image_embedding,inter_feature,img_name):
                img_emb_tensor = img_emb_tensor.cpu().clone()
                inner_emb_tensor = inner_emb_tensor.cpu().clone()
                part_dir = save_name.split('_')[0]
                tensor_save_dir = f'/x22201018/datasets/MedicalDatasets/PanNuke/{part_dir}/img_tensor'
                os.makedirs(tensor_save_dir, exist_ok=True)
                # torch.save(img_emb_tensor, f'{tensor_save_dir}/{save_name}.pt')
                torch.save(inner_emb_tensor, f'{tensor_save_dir}/{save_name}_inner_0.pt')
        
        end = time.time()
        m,s = (end-start)//60,(end-start)%60
        print(f'cost time: {int(m)}m{int(s)}s')


if __name__ == '__main__':
    gene_sam_tensor()
    # gene_mask_png()