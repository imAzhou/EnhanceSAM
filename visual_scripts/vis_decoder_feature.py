import torch
import os
import argparse
from utils import set_seed,show_mask
from models.two_stage_seg import TwoStageNet
from datasets.panoptic.create_loader import gene_loader_trainval
from mmengine.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

def onehot2instmask(onehot_mask):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
    
    return instmask

def main(logger_name, cfg):
    # load datasets
    train_dataloader, val_dataloader, metainfo, restinfo = gene_loader_trainval(
        dataset_config = d_cfg, seed = args.seed)

    # register model
    sam_model = TwoStageNet(
                num_classes = -1,
                num_mask_tokens = 4,
                sm_depth = 0,
                use_inner_feat = False,
                use_embed = True,
                sam_ckpt = cfg.sam_ckpt,
                device = device
            ).to(device)
    model = TwoStageNet(
                num_classes = cfg.num_classes,
                num_mask_tokens = cfg.num_mask_tokens,
                sm_depth = cfg.semantic_module_depth,
                use_inner_feat = cfg.use_inner_feat,
                use_embed = True,
                sam_ckpt = cfg.sam_ckpt,
                device = device
            ).to(device)
    
    model.eval()
    model.load_parameters('logs/coarse/AJI/monuseg_p256/ours_sm2_innerf0/checkpoints/best.pth')
    len_loader = len(train_dataloader)
    for i_batch, sampled_batch in enumerate(tqdm(train_dataloader, ncols=70)):

        with torch.no_grad():
            sam_outputs = sam_model.forward_coarse(sampled_batch, cfg.train_prompt_type)
            sam_pred_logits = sam_outputs['logits_256']  # (bs, 3, 256, 256)
            sam_img_embed_64 = sam_outputs['src']  # (bs, 256, 64, 64)
            sam_img_embed_256 = sam_outputs['upscaled_embedding']  # (bs, 32, 256, 256)

            outputs = model.forward_coarse(sampled_batch, cfg.train_prompt_type)
            pred_logits = outputs['logits_256']
            img_embed_64 = outputs['src']  # (bs, 256, 64, 64)
            img_embed_256 = outputs['upscaled_embedding']  # (bs, 32, 256, 256)
        
        for idx, data_sample in enumerate(sampled_batch['data_samples']):
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask = onehot2instmask(gt_inst_seg)    # tensor, (h, w)

            image_path = data_sample.img_path
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fig = plt.figure(figsize=(16,8))
            ax = fig.add_subplot(241)
            ax.imshow(image)
            show_mask(gt_sem_seg.cpu(), ax)
            ax.set_title('gt mask')

            h,w = 256,256
            show_color_gt = np.zeros((h,w,4))
            show_color_gt[:,:,3] = 1
            inst_nums = len(np.unique(gt_inst_mask)) - 1
            ax = fig.add_subplot(242)
            for i in range(inst_nums):
                color_mask = np.concatenate([np.random.random(3), [1]])
                show_color_gt[gt_inst_mask==i+1] = color_mask
            ax.imshow(show_color_gt)
            ax.set_title('color gt')

            ax = fig.add_subplot(243)
            sam_embed_256 = torch.mean(sam_img_embed_256[idx], dim=0).cpu().numpy()
            ax.imshow(sam_embed_256, cmap='hot')
            ax.set_title('sam upscaled img embed')

            # ax = fig.add_subplot(523)
            # embed_64 = torch.mean(img_embed_64[idx], dim=0).cpu().numpy()
            # ax.imshow(embed_64, cmap='hot')
            # ax.set_title('img embed after transformer')
        
            ax = fig.add_subplot(244)
            embed_256 = torch.mean(img_embed_256[idx], dim=0).cpu().numpy()
            ax.imshow(embed_256, cmap='hot')
            ax.set_title('our upscaled img embed')

            # ax = fig.add_subplot(525)
            # sam_embed_64 = torch.mean(sam_img_embed_64[idx], dim=0).cpu().numpy()
            # ax.imshow(sam_embed_64, cmap='hot')
            # ax.set_title('img embed after transformer')

            ax = fig.add_subplot(245)
            sam_large_logits = sam_pred_logits[idx][2].cpu().numpy()
            ax.imshow(sam_large_logits, cmap='RdBu')
            ax.set_title('sam large token logits')
        
            ax = fig.add_subplot(246)
            sam_medium_logits = sam_pred_logits[idx][1].cpu().numpy()
            ax.imshow(sam_medium_logits, cmap='RdBu')
            ax.set_title('sam medium logits')

            ax = fig.add_subplot(247)
            sam_small_logits = sam_pred_logits[idx][0].cpu().numpy()
            ax.imshow(sam_small_logits, cmap='RdBu')
            ax.set_title('sam small logits')
        
            ax = fig.add_subplot(248)
            our_pred_logits = pred_logits[idx][0].cpu().numpy()
            ax.imshow(our_pred_logits, cmap='RdBu')
            ax.set_title('our pred logits')

            plt.tight_layout()
            save_dir = 'visual_results/image_embed'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/{image_name}')
            plt.close()
    


if __name__ == "__main__":
    device = torch.device(args.device)

    d_cfg = Config.fromfile(args.dataset_config_file)
    model_strategy_config_file = 'configs/model_strategy.py'
    m_s_cfg = Config.fromfile(model_strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, m_s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    search_lr = cfg.get('search_lr', None)
    search_loss_type = cfg.get('search_loss_type', None)
    for loss_type in search_loss_type:
        cfg.loss_type = loss_type
        for idx,base_lr in enumerate(search_lr):
            cfg.base_lr = base_lr
            set_seed(args.seed)
            logger_name = f'lr_{loss_type}_{idx}'
            main(logger_name, cfg)



'''
python visual_scripts/vis_decoder_feature.py \
    configs/datasets/monuseg.py
'''
