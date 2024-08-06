import os
import torch
import argparse
from utils import set_seed,draw_semantic_pred
from models.two_stage_seg import TwoStageNet
from datasets.panoptic.create_loader import gene_loader_eval
from mmengine.config import Config
from utils.metrics import get_metrics
from tqdm import tqdm
import numpy as np
from mmdet.evaluation.functional import INSTANCE_OFFSET
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
        # if i_batch > 5:
        #     break
        bs_gt_mask = []
        origin_h,origin_w = sampled_batch['data_samples'][0].ori_shape
        for data_sample in sampled_batch['data_samples']:
            gt_sem_seg = data_sample.gt_sem_seg.sem_seg
            bs_gt_mask.append(gt_sem_seg)

        bs_gt_mask = torch.cat(bs_gt_mask, dim = 0).to(device)
        outputs = model.forward_coarse(sampled_batch, cfg.val_prompt_type)
        logits = outputs['logits_origin']  # (bs or k_prompt, 1, 256, 256)
        bs_pred_origin = (logits>0).detach().type(torch.uint8)

        all_datainfo = []
        for idx,datainfo in enumerate(sampled_batch['data_samples']):
            datainfo.pred_sem_seg = PixelData(sem_seg=bs_pred_origin[idx])
            
            if 'inst' in args.metrics or 'panoptic' in args.metrics:
                gt_inst_seg = datainfo.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
                gt_inst_mask = onehot2instmask(gt_inst_seg)    # tensor, (h, w)
                pred_cell_mask = (logits[idx,0,:,:].detach() > 0).type(torch.uint8)
                pred_inst_mask = evaluators['inst_evaluator'].find_connect(pred_cell_mask)
                if 'inst' in args.metrics:
                    evaluators['inst_evaluator'].process(gt_inst_mask, pred_inst_mask)
                panoptic_seg = torch.full((origin_h,origin_w), 0, dtype=torch.int32, device=device)
                if np.sum(pred_inst_mask) > 0:
                    iid = np.unique(pred_inst_mask)
                    for i in iid[1:]:
                        mask = pred_inst_mask == i
                        panoptic_seg[mask] = (1 + i * INSTANCE_OFFSET)
                datainfo.pred_panoptic_seg = PixelData(sem_seg=panoptic_seg[None])

            all_datainfo.append(datainfo.to_dict())
            if args.visual_pred and i_batch % args.visual_interval == 0:
                draw_semantic_pred(datainfo, bs_pred_origin[idx][0], pred_save_dir)
        
        evaluators['semantic_evaluator'].process(None, all_datainfo)
        if 'panoptic' in args.metrics:
            evaluators['panoptic_evaluator'].process(None, all_datainfo)
    
    total_datas = len(val_dataloader)*cfg.val_bs
    semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
    metrics = dict(
        semantic = semantic_metrics,
    )
    
    if 'inst' in args.metrics:
        inst_metrics = evaluators['inst_evaluator'].evaluate(total_datas)
        metrics['inst'] = inst_metrics
    if 'panoptic' in args.metrics:
        panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
        metrics['panoptic'] = panoptic_metrics
    return metrics



if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    
    # load datasets
    val_dataloader, metainfo, restinfo = gene_loader_eval(
        dataset_config = cfg, seed = args.seed)

    # register model
    model = TwoStageNet(
        num_classes = cfg.num_classes,
        num_mask_tokens = cfg.num_mask_tokens,
        sm_depth = cfg.semantic_module_depth,
        use_inner_feat = cfg.use_inner_feat,
        use_embed = cfg.dataset.load_embed,
        sam_ckpt = cfg.sam_ckpt,
        sam_type = cfg.sam_type,
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
    result_metrics = val_one_epoch()
    print(result_metrics)


'''
python scripts/panoptic/binary/coarse_eval.py \
    logs/0_inria/config.py \
    logs/0_inria \
    logs/0_inria/checkpoints/best.pth \
    --metrics pixel \
    --visual_pred \
    --visual_interval 40 \
'''
