import os
import torch
import argparse
from utils import set_seed,fetch_proposal_points,draw_panoptic_pred,draw_semantic_pred
from utils.sam_postprocess import process_batch_points,process_img_points,format2panoptic,format2AJI
from models.two_stage_seg import TwoStageNet
from models.sam.build_sam import sam_model_registry
from datasets.panoptic.create_loader import gene_loader_eval
from mmengine.config import Config
from utils.metrics import get_metrics
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from mmdet.evaluation.functional import INSTANCE_OFFSET
from models.sam.utils.amg import MaskData, batch_iterator
from mmengine.structures import PixelData

parser = argparse.ArgumentParser()

# base args
parser.add_argument('config_file', type=str)
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--metrics', nargs='*', type=str, 
                    default=['pixel', 'inst'], 
                    help='when metric is panoptic, val_bs must equal 1')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--points_per_batch', type=int, default=256)
parser.add_argument('--visual_pred', action='store_true')
parser.add_argument('--visual_interval', type=int, default=50)
parser.add_argument('--save_result', action='store_true')

args = parser.parse_args()

def onehot2instmask(onehot_mask):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
    
    return instmask

def val_one_epoch():
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols=70)):
        bs_gt_mask,origin_size = [],None
        for data_sample in sampled_batch['data_samples']:
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            bs_gt_mask.append(gt_sem_seg)
            origin_size = data_sample.ori_shape
        scale_ratio = 1024 // origin_size[0]

        bs_gt_mask = torch.cat(bs_gt_mask, dim = 0).to(device)
        outputs = model.forward_coarse(sampled_batch, cfg.val_prompt_type)
        logits = outputs['logits_256']  # (bs or k_prompt, 1, 256, 256)

        # calc iou metrics
        if 'pixel' in args.metrics:
            bs_pred_mask = (logits>0).detach()
            bs_pred_origin = F.interpolate(bs_pred_mask.type(torch.uint8), origin_size, mode="nearest")
            all_datainfo = []
            for idx,datainfo in enumerate(sampled_batch['data_samples']):
                
                datainfo.pred_sem_seg = PixelData(sem_seg=bs_pred_origin[idx])
                all_datainfo.append(datainfo.to_dict())

                if args.visual_pred and i_batch % args.visual_interval == 0:
                    prompts = outputs['prompts'][idx]
                    boxes,points = prompts['boxes'],prompts['points']
                    if boxes is not None:
                        boxes = boxes.squeeze(1).cpu() // scale_ratio  # (k,4)
                    if points is not None:
                        points = points[0].squeeze(0).cpu() // scale_ratio  # (k,2)

                    vis_mask = bs_pred_origin[idx].cpu()
                    pred_save_dir_seman = f'{pred_save_dir}/semantic'
                    os.makedirs(pred_save_dir_seman, exist_ok=True)
                    draw_semantic_pred(datainfo, vis_mask, pred_save_dir_seman, points, boxes)
            
            evaluators['semantic_evaluator'].process(None, all_datainfo)

        if 'inst' in args.metrics:
            assert cfg.val_bs == 1, 'metric is inst, val_bs must equal 1'
            data_sample = sampled_batch['data_samples'][0]
            gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask = onehot2instmask(gt_inst_seg)    # tensor, (h, w)
            pred_logits = logits.detach().cpu().squeeze(0).squeeze(0)    # tensor, (h, w)
            evaluators['inst_evaluator'].process(gt_inst_mask, pred_logits)

            # calc paoptic or detection metric
            # bs_image_embedding,_ = model.gene_img_embed(sampled_batch)
            
            # datainfo = sampled_batch['data_samples'][0]
            # dict_datainfo = {}
            # for k, v in datainfo.all_items():
            #     dict_datainfo[k] = v
            
            # logits_256_gray = logits.detach().cpu()
            # logits_256_gray = logits_256_gray[0].squeeze(0)
            # valued_points,edge_intensity,valued_bboxes = fetch_proposal_points(logits_256_gray)
            
            # valued_points = np.array(valued_points) * scale_ratio     # (k,2)
            # pb = args.points_per_batch
            # batch_points = batch_iterator(pb, valued_points)

            # data = MaskData()
            # for (points,) in batch_points:
            #     batch_data = process_batch_points(sam_model, points, bs_image_embedding, device, True)
            #     data.cat(batch_data)
            #     del batch_data
            
            # data = process_img_points(data)
            # panoptic_seg = format2panoptic(data, device)
            # pan_results = dict(sem_seg=panoptic_seg[None])

            pred_inst_mask = evaluators['inst_evaluator'].find_connect(pred_logits)
            panoptic_seg = torch.full((256, 256), 0, dtype=torch.int32, device=device)
            if np.sum(pred_inst_mask) > 0:
                iid = np.unique(pred_inst_mask)
                for i in iid[1:]:
                    mask = pred_inst_mask == i
                    panoptic_seg[mask] = (1 + i * INSTANCE_OFFSET)
            data_sample.pred_panoptic_seg = PixelData(sem_seg=panoptic_seg[None])
            # dict_datainfo['pred_instances'] = ins_results
            # evaluators['detection_evaluator'].process(None, [dict_datainfo])
            evaluators['panoptic_evaluator'].process(None, [data_sample.to_dict()])
            
            # if len(data.items()) > 0 and args.visual_pred and i_batch % args.visual_interval == 0:
            #     masks = np.array(data["segmentations"])     # np.array, (k, h, w) size in original image
            #     merged_masks = np.max(masks, axis=0)     # np.array, (h, w) 
            #     pred_save_dir_panop = f'{pred_save_dir}/panoptic'
            #     os.makedirs(pred_save_dir_panop, exist_ok=True)
            #     draw_panoptic_pred(dict_datainfo, {
            #         'bboxes': data['boxes'],
            #         'boxes_clsids': np.ones((data['boxes'].shape[0], ), dtype=np.int32),
            #         'masks': merged_masks,
            #     }, logits_256_gray, edge_intensity,valued_bboxes, metainfo, pred_save_dir_panop)

    total_datas = len(val_dataloader)*cfg.val_bs
    semantic_metrics,panoptic_metrics = None,None
    if 'pixel' in args.metrics:
        semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
    if 'inst' in args.metrics:
        # detection_metrics = evaluators['detection_evaluator'].evaluate(total_datas)
        panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
        inst_metrics = evaluators['inst_evaluator'].evaluate(total_datas)
    
    return semantic_metrics,panoptic_metrics,inst_metrics


if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    
    # load datasets
    val_dataloader, metainfo, restinfo = gene_loader_eval(
        dataset_config = cfg, seed = args.seed)

    # register model
    sam_model = sam_model_registry['vit_h'](checkpoint = cfg.sam_ckpt).to(device)
    model = TwoStageNet(
        num_classes = cfg.num_classes,
        num_mask_tokens = cfg.num_mask_tokens,
        sm_depth = cfg.semantic_module_depth,
        use_inner_feat = cfg.use_inner_feat,
        use_embed = True,
        sam_ckpt = cfg.sam_ckpt,
        device = device
    ).to(device)
    
    if args.visual_pred:
        pred_save_dir = f'{args.result_save_dir}/vis_pred'
        os.makedirs(pred_save_dir, exist_ok = True)
    if args.save_result:
        result_dir = f'{args.result_save_dir}/results'
        os.makedirs(result_dir, exist_ok = True)

    # get evaluator
    evaluators = get_metrics(args.metrics, metainfo, restinfo)

    # val
    model.load_parameters(args.ckpt_path)
    semantic_metrics, panoptic_metrics,inst_metrics = val_one_epoch()
    if semantic_metrics is not None:
        print(semantic_metrics)
    if panoptic_metrics is not None:
        print(panoptic_metrics)
    if inst_metrics is not None:
        print(inst_metrics)


'''
python scripts/panoptic/binary/coarse_eval.py \
    logs/coarse/monuseg_p256/inner_f/config.py \
    logs/coarse/monuseg_p256/inner_f/debug \
    logs/coarse/monuseg_p256/inner_f/checkpoints/best_miou.pth \
    --metrics pixel inst \
    --visual_pred \
    --visual_interval 3
    --points_per_batch 256 \
    --visual_pred
'''
