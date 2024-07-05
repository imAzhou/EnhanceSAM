import os
import torch
import argparse
from utils import set_seed,fetch_proposal_points,show_mask,show_box
from models.two_stage_seg import TwoStageNet
import json
from datasets.panoptic.create_loader import gene_loader_eval
from mmengine.config import Config
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual_salient_p', action='store_true')
parser.add_argument('--visual_interval', type=int, default=50)

args = parser.parse_args()

def draw_salient_p(logits_gray, edge_intensity, bboxes, sampled_batch, save_dir):
    datainfo = sampled_batch['data_samples'][0]
    image_path = datainfo.img_path
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    origin_mask = datainfo.gt_sem_seg.sem_seg  # (1, h,w)

    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(131)
    ax.imshow(image)
    show_mask(origin_mask.cpu(), ax)
    ax.set_title('gt mask')
    ax.set_axis_off()

    ax = fig.add_subplot(132)
    ax.imshow(logits_gray, cmap='gray')
    ax.set_title('logits gray')
    ax.set_axis_off()

    edge_intensity*=255
    ax = fig.add_subplot(133)
    ax.imshow(edge_intensity, cmap='gray')
    for box in bboxes:
        show_box(box, ax)
    ax.set_title('edge intensity used valued boxes')
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{image_name}')
    plt.close()

def val_one_epoch():
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols=70)):
        data_sample = sampled_batch['data_samples'][0]
        img_path = data_sample.img_path
        img_dir,img_name = os.path.dirname(img_path),os.path.basename(img_path)
        json_save_dir = img_dir.replace('img_dir','salient_p')
        os.makedirs(json_save_dir, exist_ok=True)
        json_save_path = f'{json_save_dir}/{img_name}'.replace('.png','.json')
        origin_size = data_sample.ori_shape
        scale_ratio = 1024 // origin_size[0]

        outputs = model.forward_coarse(sampled_batch, cfg.val_prompt_type)
        logits = outputs['logits_256']  # (bs or k_prompt, 1, 256, 256)
        logits_256_gray = logits.detach().cpu()
        logits_256_gray = logits_256_gray[0].squeeze(0)
        valued_points,edge_intensity,valued_bboxes = fetch_proposal_points(logits_256_gray)
        
        salient_dict = dict(
            img_path = img_path,
            scale_ratio = scale_ratio,
            point_type = 'xy',
            points = (np.array(valued_points) * scale_ratio).tolist()
        )

        with open(json_save_path, 'w') as json_f:
            json.dump(salient_dict, json_f)

        if args.visual_salient_p and i_batch % args.visual_interval == 0:
            visua_save_dir = img_dir.replace('img_dir','salient_p_visual')
            os.makedirs(visua_save_dir, exist_ok=True)
            draw_salient_p(logits_256_gray, edge_intensity, valued_bboxes, sampled_batch, visua_save_dir)
        

if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    assert cfg.val_bs == 1, 'val_bs must equal 1'
    
    # load datasets
    val_dataloader, metainfo, restinfo = gene_loader_eval(
        dataset_config = cfg, seed = args.seed)

    # register model
    model = TwoStageNet(
        num_classes = 1,
        sm_depth = cfg.semantic_module_depth,
        use_inner_feat = cfg.use_inner_feat,
        use_embed = True,
        sam_ckpt = cfg.sam_ckpt,
        device = device
    ).to(device)

    # val
    model.load_parameters(args.ckpt_path)
    val_one_epoch()
    

'''
python scripts/panoptic/salient_p_prestore.py \
    logs/coarse/monuseg_p256/sm_2_inner_f/config.py \
    logs/coarse/monuseg_p256/sm_2_inner_f/checkpoints/epoch_11.pth \
    --visual_salient_p \
    --visual_interval 20
'''
