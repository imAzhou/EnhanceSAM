import torch
import os
import argparse
from utils import set_seed
from models.two_stage_seg import TwoStageNet
from datasets.panoptic.create_loader import gene_loader_trainval
from mmengine.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion

parser = argparse.ArgumentParser()

# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt_path', type=str, default='cuda:0')
parser.add_argument('--save_interval', type=int, default=20, help='random seed')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

def onehot2instmask(onehot_mask, with_boundary=False):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    inst_boundary_mask = np.zeros((len(onehot_mask), h, w))
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
        if with_boundary:
            dilated_mask = binary_dilation(seg)
            eroded_mask = binary_erosion(seg)
            boundary_mask = dilated_mask ^ eroded_mask   # binary mask (h,w)
            inst_boundary_mask[iid] = boundary_mask
    
    if with_boundary:
        inst_overlap_boundary_mask = np.sum(inst_boundary_mask, axis=0)
        inst_overlap_boundary_mask = (inst_overlap_boundary_mask > 1).astype(int)
        inst_overlap_boundary_mask = binary_dilation(inst_overlap_boundary_mask, iterations=2)
        inst_boundary_mask = np.max(inst_boundary_mask, axis=0)
        inst_boundary_mask[inst_overlap_boundary_mask] = 1
        return instmask, inst_boundary_mask
    
    return instmask

def find_connect(binary_mask: torch.Tensor, dilation = False):

    image_mask_np = binary_mask.cpu().numpy().astype(np.uint8)
    # labels: 0 mean background, 1...n mean different connection region
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=4)
    if dilation and np.sum(labels) > 0:
        iid = np.unique(labels)
        for i in iid[1:]:
            mask = labels == i
            dilation_mask = binary_dilation(mask)
            labels[dilation_mask] = i
    
    return labels

def draw_semantic(data_sample, draw_items, save_dir):
    image_path = data_sample.img_path
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_sem_seg = data_sample.gt_sem_seg.sem_seg

    fig = plt.figure(figsize=(8,16))
    ax = fig.add_subplot(321)
    ax.imshow(image)
    ax.set_title('image')

    gt_binary_mask = (gt_sem_seg*255).squeeze(0).numpy()
    ax = fig.add_subplot(322)
    ax.imshow(gt_binary_mask, cmap='gray')
    ax.set_title('gt semantic mask')

    ax = fig.add_subplot(323)
    sam_img_embed_256 = draw_items['sam_img_embed_256']
    sam_embed_256 = torch.mean(sam_img_embed_256, dim=0).cpu().numpy()
    ax.imshow(sam_embed_256, cmap='hot')
    ax.set_title('sam upscaled img embed')

    ax = fig.add_subplot(324)
    sam_pred_mask = draw_items['sam_pred_semantic']
    ax.imshow(sam_pred_mask.cpu().numpy(), cmap='gray')
    ax.set_title('sam pred semantic mask')

    ax = fig.add_subplot(325)
    our_img_embed_256 = draw_items['our_img_embed_256']
    embed_256 = torch.mean(our_img_embed_256, dim=0).cpu().numpy()
    ax.imshow(embed_256, cmap='hot')
    ax.set_title('our upscaled img embed')

    ax = fig.add_subplot(326)
    our_pred_mask = draw_items['our_pred_semantic']
    ax.imshow(our_pred_mask.cpu().numpy(), cmap='gray')
    ax.set_title('sam pred semantic mask')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{image_name}')
    plt.close()

def get_color_inst_mask(inst_mask):
    h,w = inst_mask.shape
    show_color_mask = np.zeros((h,w,4))
    show_color_mask[:,:,3] = 1    # opacity
    inst_nums = len(np.unique(inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_mask[inst_mask==i+1] = color_mask
    
    return show_color_mask

def draw_instance(data_sample, draw_items, save_dir):
    image_path = data_sample.img_path
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _,h,w = image.shape
    gt_sem_seg = data_sample.gt_sem_seg.sem_seg

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(331)
    ax.imshow(image)
    ax.set_title('image')
    gt_binary_mask = (gt_sem_seg*255).squeeze(0).numpy()
    ax = fig.add_subplot(332)
    ax.imshow(gt_binary_mask, cmap='gray')
    ax.set_title('gt semantic mask')
    ax = fig.add_subplot(333)
    gt_inst_mask = draw_items['gt_inst_mask']
    gt_color_mask = get_color_inst_mask(gt_inst_mask)
    ax.imshow(gt_color_mask)
    ax.set_title('gt instance mask')

    ax = fig.add_subplot(334)
    sam_img_embed_256 = draw_items['sam_img_embed_256']
    sam_embed_256 = torch.mean(sam_img_embed_256, dim=0).cpu().numpy()
    ax.imshow(sam_embed_256, cmap='hot')
    ax.set_title('sam upscaled img embed')
    ax = fig.add_subplot(335)
    gt_boundary_mask = draw_items['gt_boundary_mask']
    ax.imshow(gt_boundary_mask, cmap='gray')
    ax.set_title('gt boundary mask')
    ax = fig.add_subplot(336)
    pred_boundary_mask = draw_items['pred_boundary_mask']
    ax.imshow(pred_boundary_mask, cmap='gray')
    ax.set_title('sam pred instance mask')

    ax = fig.add_subplot(337)
    our_img_embed_256 = draw_items['our_img_embed_256']
    our_embed_256 = torch.mean(our_img_embed_256, dim=0).cpu().numpy()
    ax.imshow(our_embed_256, cmap='hot')
    ax.set_title('our upscaled img embed')
    ax = fig.add_subplot(338)
    our_pred_semantic = draw_items['our_pred_semantic']
    ax.imshow(our_pred_semantic.cpu().numpy(), cmap='gray')
    ax.set_title('our pred semantic mask')
    ax = fig.add_subplot(339)
    our_pred_instance = draw_items['our_pred_instance']
    our_color_mask = get_color_inst_mask(our_pred_instance)
    ax.imshow(our_color_mask)
    ax.set_title('our pred instance mask')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{image_name}')
    plt.close()


def main():
    # load datasets
    train_dataloader, val_dataloader, metainfo, restinfo = gene_loader_trainval(
        dataset_config = cfg, seed = args.seed)
    
    dataset_tag = cfg['dataset_tag']
    save_dir = f'visual_results/embed_with_sema_inst/{dataset_tag}'

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
                use_embed = cfg.dataset.load_embed,
                use_boundary_head = cfg.use_boundary_head,
                sam_ckpt = cfg.sam_ckpt,
                device = device
            ).to(device)
    
    model.eval()
    model.load_parameters(args.ckpt_path)
    for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols=70)):

        with torch.no_grad():
            sam_outputs = sam_model.forward_coarse(sampled_batch)
            sam_pred_logits = sam_outputs['logits_256']  # (bs, 1, 256, 256)
            sam_img_embed_256 = sam_outputs['upscaled_embedding']  # (bs, 32, 256, 256)

            outputs = model.forward_coarse(sampled_batch)
            pred_logits = outputs['logits_256']
            img_embed_256 = outputs['upscaled_embedding']  # (bs, 32, 256, 256)
        
        for idx, data_sample in enumerate(sampled_batch['data_samples']):
            
            gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask,gt_boundary = onehot2instmask(gt_inst_seg, with_boundary=True)

            sam_pred_fore_mask = (F.sigmoid(sam_pred_logits[idx,0,:,:]).detach() > 0.).type(torch.uint8)
            pred_fore_mask = (pred_logits[idx,0,:,:].detach() > 0).type(torch.uint8)
            draw_items = {
                'sam_img_embed_256': sam_img_embed_256[idx],
                'sam_pred_semantic': sam_pred_fore_mask,
                'our_img_embed_256': img_embed_256[idx],
                'our_pred_semantic': pred_fore_mask
            }
            if cfg.use_boundary_head:
                pred_boundary_mask = pred_logits[idx,1,:,:].detach() > 0
                pred_fore_mask[pred_boundary_mask] = 0
                pred_inst_mask = find_connect(pred_fore_mask, dilation = True)
                pred_fore_mask = torch.as_tensor(pred_inst_mask > 0).type(torch.uint8)
                sam_pred_inst_mask = find_connect(sam_pred_fore_mask, dilation = False)

                draw_items.update({
                    'gt_inst_mask': gt_inst_mask,
                    'sam_pred_instance': sam_pred_inst_mask,
                    'our_pred_semantic': pred_fore_mask,
                    'our_pred_instance': pred_inst_mask,
                    'gt_boundary_mask': gt_boundary,
                    'pred_boundary_mask': pred_boundary_mask.cpu(),
                })

            if i_batch % args.save_interval == 0:
                if cfg.use_boundary_head:
                    draw_instance(data_sample, draw_items, save_dir)
                else:
                    draw_semantic(data_sample, draw_items, save_dir)
            

if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    
    main()



'''
python visual_scripts/vis_decoder_feature.py \
    logs/0_cpm17/config.py \
    logs/0_cpm17/checkpoints/best.pth \
    --save_interval 10
'''
