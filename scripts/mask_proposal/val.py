import time
import os
import torch
import argparse
from help_func.build_sam import sam_model_registry
from help_func.tools import set_seed
from models.mask_proposal import MaskProposalNet
from tqdm import tqdm
from datasets.loveda import LoveDADataset
from torch.utils.data import DataLoader
from utils.iou_metric import IoUMetric
from utils.local_visualizer import SegLocalVisualizer
import mmcv
from utils.data_structure import SegDataSample
from mmengine.structures import PixelData

parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type=str)
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--num_queries', type=int,
                    default=100, help='output masks of network')
# parser.add_argument('--point_num', type=int,
#                     default=1000, help='output masks of network')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--sam_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--visual', action='store_true', help='If activated, the predict mask will be saved')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = f'logs/mask_proposal/{args.dir_name}'
    
    # register model
    sam = sam_model_registry['vit_h'](image_size = args.img_size).to(device)
    model = MaskProposalNet(sam,
                    num_classes = args.num_classes,
                    num_queries = args.num_queries,
                    # point_num = args.point_num,
                    sam_ckpt_path=args.sam_ckpt
                ).to(device)
    model.eval()
    
    # load datasets
    dataset = LoveDADataset(
        data_root = '/x22201018/datasets/RemoteSensingDatasets/LoveDA',
        resize_size = args.img_size,
        mode = 'val',
        use_embed = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    
    test_evaluator = IoUMetric(iou_metrics=['mIoU'])
    test_evaluator.dataset_meta = dataset.METAINFO
    seg_local_visualizer = SegLocalVisualizer(
        save_dir = f'{record_save_dir}/pred_vis',
        classes = dataset.METAINFO['classes'],
        palette = dataset.METAINFO['palette'],
    )
    
    all_metrics = []
    all_miou = []
    max_iou,max_epoch = 0,0
    for epoch_num in range(args.max_epochs):
    # for epoch_num in range(14,15):
        pth_load_path = f'{record_save_dir}/checkpoints/epoch_{epoch_num}.pth'
        model.load_parameters(pth_load_path)
        for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
            image_batch, label_batch = sampled_batch['image_tensor'].to(device), sampled_batch['gt_mask'].long().to(device)
            gt_origin_masks = sampled_batch['gt_origin_mask']
            im_original_size = (sampled_batch['original_size'][0].item(),sampled_batch['original_size'][1].item())
            im_input_size = sampled_batch['input_size']

            bs_image_embedding = sampled_batch['img_embed'].to(device)
            outputs = model.predict(bs_image_embedding, label_batch)
            
            # shape: [num_classes, 1024, 1024]
            pred_logits = model.postprocess_masks(
                outputs['pred_logits'],
                input_size=im_input_size,
                original_size=im_original_size
            ).squeeze(0)
            # pred_sem_seg.shape: (1, H, W)
            pred_sem_seg = pred_logits.argmax(dim=0, keepdim=True).cpu()

            test_evaluator.process(pred_sem_seg, gt_origin_masks)

            # if args.visual and i_batch % 50 == 0:
            if args.visual:
                # 可视化时过滤掉背景区域的预测结果(将pred中背景区域的值也置为255，可视化时就会被忽略)
                bg_mask = (gt_origin_masks == 255)
                pred_sem_seg[bg_mask] = 255
                
                origin_image = mmcv.imread(sampled_batch['meta_info']['img_path'][0])
                data_sample = SegDataSample()
                data_sample.set_data({
                    'gt_sem_seg': PixelData(**{'data': gt_origin_masks}),
                    'pred_sem_seg': PixelData(**{'data': pred_sem_seg}),
                })
                seg_local_visualizer.add_datasample(sampled_batch['meta_info']['img_name'][0], origin_image, data_sample)

        metrics = test_evaluator.evaluate(len(dataloader))
        print(f'epoch: {epoch_num}' + str(metrics) + '\n')
        all_miou.append(metrics['mIoU'])
        if metrics['mIoU'] > max_iou:
            max_iou = metrics['mIoU']
            max_epoch = epoch_num
        all_metrics.append(f'epoch: {epoch_num}' + str(metrics) + '\n')
    # save result file
    config_file = os.path.join(record_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_metrics)
        f.write(f'\nmax_iou: {max_iou}, max_epoch: {max_epoch}\n')
        f.write(str(all_miou))

'''
python scripts/mask_proposal/val.py \
    --dir_name 2024_01_12_11_20_52 \
    --batch_size 1 \
    --num_classes 7 \
    --max_epochs 24 \
    --point_num 10 \
    --visual
'''
