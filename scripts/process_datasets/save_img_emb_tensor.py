import time
import os
import torch
from models.sam.build_sam import sam_model_registry
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
from datasets.loveda import LoveDADataset
from torch.utils.data import DataLoader
from utils import set_seed

if __name__ == "__main__":
    set_seed(1234)
    device = torch.device('cuda:0')
    image_size = 1024
    batch_size = 1
    # data_root = '/x22201018/datasets/RemoteSensingDatasets'
    data_root = '/nfs/zly/datasets'
    # dataset_name = 'WHU-Building'   # WHU-Building  InriaBuildingDataset
    dataset_name = 'LoveDA'

    # register model
    sam = sam_model_registry['vit_h'](image_size = image_size,
                                        checkpoint = 'checkpoints_sam/sam_vit_h_4b8939.pth',
                                        # checkpoint = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth',
                                    ).to(device)
    sam.eval()

    for mode in ['train','val']:
    # for mode in ['val']:
        tensor_save_dir = f'{data_root}/{dataset_name}/img_dir/{mode}_tensor'
        os.makedirs(tensor_save_dir, exist_ok=True)
        # load datasets
        # val_dataset = BuildingDataset(
        #     data_root = f'{data_root}/{dataset_name}',
        #     mode = mode,
        # )
        # dataloader = DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=8,
        #     drop_last=True)
        dataset = LoveDADataset(
            data_root = f'{data_root}/{dataset_name}',
            mode = mode
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False)

        start = time.time()
        for batch_idx, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
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
        
        end = time.time()
        m,s = (end-start)//60,(end-start)%60
        print(f'cost time: {int(m)}m{int(s)}s')
