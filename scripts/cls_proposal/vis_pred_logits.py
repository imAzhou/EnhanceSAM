import os
import torch
import argparse
from help_func.build_sam import sam_model_registry
from models.cls_proposal import ClsProposalNet
from tqdm import tqdm
from datasets.loveda import LoveDADataset
from torch.utils.data import DataLoader
from utils.local_visualizer import SegLocalVisualizer
import mmcv
from utils.data_structure import SegDataSample
from mmengine.structures import PixelData
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.visualization import show_mask

parser = argparse.ArgumentParser()

parser.add_argument('--ckp_path', type=str)
parser.add_argument('--count', type=int)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--sam_ckpt', type=str, 
                    default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')

args = parser.parse_args()

def vis_pred_logits(pred_logits, gt_masks, class_names, img_path, save_dir):

    dpi = 100
    figsize = (1024/dpi, 1024/dpi)
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)
    img = plt.imread(img_path)
    for cls_i in range(len(class_names)):
        cls_gt_mask = gt_masks.detach().cpu().squeeze(0) == cls_i
        cls_pred_logits = pred_logits[cls_i].sigmoid().detach().cpu()
        plt.imshow(img)
        plt.imshow(cls_pred_logits.numpy(), cmap="hot", alpha=0.7)
        show_mask(cls_gt_mask, plt.gca())
        plt.axis('off')
        plt.title(class_names[cls_i])
    
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{class_names[cls_i]}.png')
        plt.clf()




if __name__ == "__main__":
    num_classes = 7
    device = torch.device('cuda:0')
    
    # register model
    sam = sam_model_registry['vit_h'](image_size = 1024).to(device)
    model = ClsProposalNet(sam,
                    num_classes = num_classes,
                    sam_ckpt_path=args.sam_ckpt
                ).to(device)
    model.eval()
    
    # load datasets
    dataset = LoveDADataset(
        data_root = '/x22201018/datasets/RemoteSensingDatasets/LoveDA',
        resize_size = 1024,
        mode = 'val',
        use_embed = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 4,
        drop_last = True)
    
    seg_local_visualizer = SegLocalVisualizer(
        save_dir = f'{args.save_dir}/pred_vis',
        classes = dataset.METAINFO['classes'],
        palette = dataset.METAINFO['palette'],
    )
    
    pth_load_path = args.ckp_path
    model.load_parameters(pth_load_path)
    for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
        if i_batch > args.count:
            break
        image_batch, label_batch = sampled_batch['image_tensor'].to(device), sampled_batch['gt_mask'].long().to(device)
        gt_origin_masks = sampled_batch['gt_origin_mask']
        im_original_size = (sampled_batch['original_size'][0].item(),sampled_batch['original_size'][1].item())
        im_input_size = sampled_batch['input_size']
        image_path = sampled_batch['meta_info']['img_path'][0]
        image_name = sampled_batch['meta_info']['img_name'][0]

        bs_image_embedding = sampled_batch['img_embed'].to(device)
        outputs = model(bs_image_embedding)


        # shape: [num_classes, 1024, 1024]
        pred_logits = model.postprocess_masks(
            outputs['pred_logits'],
            input_size=im_input_size,
            original_size=im_original_size
        ).squeeze(0)
        vis_pred_logits(pred_logits,label_batch, dataset.METAINFO['classes'], 
                        image_path, save_dir=f'{args.save_dir}/pred_logits/{image_name}')

        # pred_sem_seg.shape: (1, H, W)
        pred_sem_seg = pred_logits.argmax(dim=0, keepdim=True).cpu()

        # 可视化时过滤掉背景区域的预测结果(将pred中背景区域的值也置为255，可视化时就会被忽略)
        bg_mask = (gt_origin_masks == 255)
        pred_sem_seg[bg_mask] = 255
        
        origin_image = mmcv.imread(image_path)
        data_sample = SegDataSample()
        data_sample.set_data({
            'gt_sem_seg': PixelData(**{'data': gt_origin_masks}),
            'pred_sem_seg': PixelData(**{'data': pred_sem_seg}),
        })
        seg_local_visualizer.add_datasample(image_name, origin_image, data_sample)




'''
python scripts/cls_proposal/vis_pred_logits.py \
    --ckp_path logs/cls_proposal/2024_01_12_11_20_52_useConvSkip/checkpoints/epoch_10.pth \
    --save_dir logs/cls_proposal/2024_01_12_11_20_52_useConvSkip \
    --count 20
'''
