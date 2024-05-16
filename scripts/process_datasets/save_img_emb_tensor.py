import time
import os
import torch
from models.sam.build_sam import sam_model_registry
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
# from datasets.semantic_seg.loveda import LoveDADataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
from utils import set_seed

if __name__ == "__main__":
    set_seed(1234)
    device = torch.device('cuda:0')
    image_size = 1024
    batch_size = 1
    data_root = '/x22201018/datasets/RemoteSensingDatasets'
    dataset_name = 'WHU-Building'   # WHU-Building  InriaBuildingDataset
    use_aug = False
    # register model
    sam = sam_model_registry['vit_h'](image_size = image_size,
                                        # checkpoint = 'checkpoints_sam/sam_vit_h_4b8939.pth',
                                        checkpoint = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth',
                                    ).to(device)
    sam.eval()

    for mode in ['train','val']:
    # for mode in ['val']:
        tail = 'aug_tensor' if use_aug else 'tensor'
        tensor_save_dir = f'{data_root}/{dataset_name}/img_dir/{mode}_{tail}'
        os.makedirs(tensor_save_dir, exist_ok=True)
        # load datasets
        val_dataset = BuildingDataset(
            data_root = f'{data_root}/{dataset_name}',
            mode = mode,
            use_aug = use_aug
        )
        dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True)
        # dataset = LoveDADataset(
        #     data_root = '/x22201018/datasets/RemoteSensingDatasets/LoveDA',
        #     resize_size = image_size,
        #     mode = mode
        # )
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=2,
        #     drop_last=False)

        start = time.time()
        for batch_idx, sampled_batch in enumerate(tqdm(dataloader)):
            image_batch = sampled_batch['input_image'].to(device)
            img_path,img_name = sampled_batch['meta_info']['img_path'],sampled_batch['meta_info']['img_name']

            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                input_images = sam.preprocess(image_batch)
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                bs_image_embedding = sam.image_encoder(input_images)
            for img_emb_tensor,save_name in zip(bs_image_embedding,img_name):
                img_emb_tensor = img_emb_tensor.cpu().clone()
                torch.save(img_emb_tensor, f'{tensor_save_dir}/{save_name}.pt')
            
                if use_aug:
                    mask_save_dir = f'datasets/{dataset_name}/ann_dir/{mode}_aug'
                    aug_img_save = f'datasets/{dataset_name}/img_dir/{mode}_aug'
                    os.makedirs(mask_save_dir, exist_ok=True)
                    os.makedirs(aug_img_save, exist_ok=True)
                    mask_1024 = (sampled_batch['mask_1024'][0]*255).numpy().astype(np.uint8)
                    cv2.imwrite(f'{mask_save_dir}/{save_name}.png', mask_1024)
                    cv2.imwrite(f'{aug_img_save}/{save_name}.png', image_batch[0].permute(1,2,0).cpu().numpy())
        
        end = time.time()
        m,s = (end-start)//60,(end-start)%60
        print(f'cost time: {int(m)}m{int(s)}s')
