import os
import torch
import argparse
from utils import set_seed,get_det, get_connection, draw_det_result,get_det_multiprocessing,draw_panoptic_pred
from models.prompt_seg import PromptSegNet
from datasets.panoptic.create_loader import gene_loader
from mmengine.config import Config
from models.sam.build_sam import sam_model_registry
from tqdm import tqdm
from mmcv.ops import nms
import numpy as np
from models.sam.utils.amg import batch_iterator
from mmdet.structures.mask import mask2bbox
from mmdet.evaluation.metrics import CocoPanopticMetric,CocoMetric
from mmengine.structures import InstanceData, PixelData
from mmdet.evaluation.functional import INSTANCE_OFFSET
import matplotlib.pyplot as plt
from utils.sam_postprocess import process_batch,postprocess_small_regions
from models.sam.utils.amg import (
    MaskData,uncrop_boxes_xyxy,uncrop_points,rle_to_mask
)
from torchvision.ops.boxes import batched_nms, box_area


parser = argparse.ArgumentParser()

# base args
# parser.add_argument('dataset_config_file', type=str)
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--prompt_type', type=str,)
parser.add_argument('--points_per_batch', type=int, default=64)


args = parser.parse_args()
thresh_center = 0.7
rad = 10

def is_box_inside(box1, box2):
    """
    判断 box1 是否被 box2 完全包围
    box: [x_min, y_min, x_max, y_max]
    """
    return (box1[0] >= box2[0] and
            box1[1] >= box2[1] and
            box1[2] <= box2[2] and
            box1[3] <= box2[3])

def filter_boxes(boxes):
    """
    过滤掉被其他 box 完全包围的 box
    boxes: numpy array of shape (N, 4), where each row is [x_min, y_min, x_max, y_max]
    """
    keep = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i != j and is_box_inside(boxes[i], boxes[j]):
                keep[i] = False
                break
    keep_idxs = np.where(keep)[0]
    return keep_idxs

def fetch_proposal_points(logits_256_gray):

    logits_256_gray[logits_256_gray<0] = 0
    valued_points,valued_bboxes = [],[]

    min_v, max_v = torch.min(logits_256_gray),torch.max(logits_256_gray)
    if max_v > 0:
        logits_256_gray = ((logits_256_gray - min_v) / (max_v - min_v)) * 255
        logits_256_gray = logits_256_gray.numpy()
        # edge_intensity = get_det(logits_256_gray)   # 每张耗时 1.2s 左右
        edge_intensity = get_det_multiprocessing(logits_256_gray)   # 每张耗时 1.2s 左右
        edge_intensity[edge_intensity>0] = 1
        avgidx, avgbboxs = get_connection(edge_intensity,thresh=1)

        # fig = plt.figure(figsize=(8,4))
        # ax = fig.add_subplot(121)
        # ax.imshow(logits_256_gray, cmap='gray')
        # ax.set_title('logits gray')
        # edge_intensity*=255
        # ax = fig.add_subplot(122)
        # ax.imshow(edge_intensity, cmap='gray')
        # ax.set_title('edge intensity')
        # plt.tight_layout()
        # plt.savefig('get_det_multiprocessing_optimize.png')
        # plt.close()
        
        valued_bboxes = []
        for avgbox in avgbboxs:
            hmin,wmin,hmax,wmax = avgbox
            center = [int((hmax+hmin)/2),int((wmax+wmin)/2)]
            if logits_256_gray[center[0], center[1]] > 0:
                valued_points.append([center[1], center[0]])
                valued_bboxes.append([wmin, hmin, wmax, hmax])
    
    return valued_points,edge_intensity,valued_bboxes

def val_one_epoch_binary(model: PromptSegNet, val_loader, panoptic_evaluator:CocoPanopticMetric, detection_evaluator):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        # if i_batch > 30:
        #     break
        outputs = model.forward_single_class(sampled_batch, args.prompt_type)
        logits_256 = outputs['logits_256']  # logits_256.shape: (bs, 1, 256, 256)

        bs = logits_256.shape[0]
        for idx in range(bs):
            datainfo = sampled_batch['data_samples'][idx]
            dict_datainfo = {}
            for k, v in datainfo.all_items():
                dict_datainfo[k] = v

            logits_256_gray = logits_256.detach().cpu()
            logits_256_gray = logits_256_gray[idx].squeeze(0)
            valued_points,edge_intensity,valued_bboxes = fetch_proposal_points(logits_256_gray)
            scale_ratio = 4
            valued_points = np.array(valued_points) * scale_ratio     # (k,2)

            pb = args.points_per_batch
            batch_points = batch_iterator(pb, valued_points)

            data = MaskData()
            for (points,) in batch_points:
                coords_torch = torch.as_tensor(points, dtype=torch.float, device=device)
                labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
                coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
                point_prompt = (coords_torch, labels_torch)

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=point_prompt,
                    boxes=None,
                    masks=None,
                )

                bs_image_embedding,_ = model.gene_img_embed(sampled_batch)
                sam_outputs = sam_model.mask_decoder(
                    image_embeddings = bs_image_embedding,
                    image_pe = sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = True
                )
                sam_logits,sam_ious = sam_outputs[0],sam_outputs[1]
                sam_pred_mask = sam_logits > 0  # (k, m, 256, 256)
                sam_pred_mask = sam_pred_mask[:, 0, ::]  # (k, 256, 256)
                sam_ious = sam_ious[:, 0]  # (k,)
                
                sam_logits_process = sam_logits[:, 0, ::].unsqueeze(1).detach()
                sam_ious_process = sam_ious.unsqueeze(1).detach()
                batch_data = process_batch(
                    points // scale_ratio, sam_logits_process, sam_ious_process,
                    orig_size = (256,256),
                    pred_iou_thresh = 0.5,
                    stability_score_thresh = 0.7
                )
                data.cat(batch_data)
                del batch_data
            
            h, w = 256,256
            bg_clsid = 0
            cell_clsid = 1
            panoptic_seg = torch.full((h, w), bg_clsid, dtype=torch.int32, device=device)

            if len(data.items()) > 0:                
                # Remove duplicates within this crop.
                box_nms_thresh = 0.5
                keep_by_nms = batched_nms(
                    data["boxes"].float(),
                    data["iou_preds"],
                    torch.zeros_like(data["boxes"][:, 0]),  # categories
                    iou_threshold=box_nms_thresh,
                )
                data.filter(keep_by_nms)

                keep_by_nocontain = filter_boxes(data["boxes"])
                data.filter(keep_by_nocontain)

                # Return to the original image frame
                crop_box = [0, 0, 256, 256]
                data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
                data["points"] = uncrop_points(data["points"], crop_box)
                data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

                data.to_numpy()
                min_mask_region_area = 0
                # Filter small disconnected regions and holes in masks
                if min_mask_region_area > 0:
                    data = postprocess_small_regions(
                        data,
                        min_mask_region_area,
                        box_nms_thresh,
                    )
                bboxes = data['boxes']     # np.array, (k,2) size in original image
                data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]
                masks = np.array(data["segmentations"])     # np.array, (k, h, w) size in original image
                if len(masks) > 0:
                    merged_masks = np.max(masks, axis=0)     # np.array, (h, w) 
                    # draw_panoptic_pred(dict_datainfo, {
                    #     'bboxes': bboxes,
                    #     'boxes_clsids': np.ones((bboxes.shape[0], ), dtype=np.int32),
                    #     'masks': merged_masks,
                    # }, logits_256_gray, edge_intensity,valued_bboxes, metainfo, det_save_dir)

                instance_id = 1
                if len(bboxes) >0:
                    for i in range(len(bboxes)):
                        mask = masks[i]
                        panoptic_seg[mask] = (cell_clsid + instance_id * INSTANCE_OFFSET)
                        instance_id += 1
            
            pan_results = dict(sem_seg=panoptic_seg[None])
            dict_datainfo['pred_panoptic_seg'] = pan_results

            # ins_results = dict(
            #     bboxes = nms_boxes,
            #     labels = torch.ones_like(nms_scores) * cell_clsid,
            #     scores = nms_scores,
            #     masks = nms_masks
            # )
            # dict_datainfo['pred_instances'] = ins_results
            

            # coords_torch = coords_torch / scale_ratio
            # point_prompt = (coords_torch, labels_torch)

        panoptic_evaluator.process(None, [dict_datainfo])

    panoptic_metrics = panoptic_evaluator.evaluate(len(val_loader))
    print(panoptic_metrics)


if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # register model
    sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
    sam_model = sam_model_registry['vit_h'](checkpoint = sam_ckpt).to(device)
    model = PromptSegNet(
                num_classes = 1,
                sm_depth = 2,
                use_inner_feat = True,
                use_multi_mlps = False,
                use_mask_prompt = False,
                use_embed = True,
                sam_ckpt = sam_ckpt,
                device = device
            ).to(device)
    
    dataset_config_file = 'configs/datasets/pannuke.py'
    d_cfg = Config.fromfile(dataset_config_file)
    dataset_config = dict(
        load_parts = ['Part3'],
        pure_args = d_cfg
    )
    dataloader_config = dict(
        batch_size = 1,
        num_workers = 16,
        seed = args.seed
    )
    dataloader,metainfo = gene_loader(
        dataset_tag = 'pannuke_binary', # or pannuke_binary
        dataset_config = dataset_config,
        dataloader_config = dataloader_config
    )
    
    det_save_dir = f'{args.result_save_dir}/vis_detection_sam_process'
    os.makedirs(det_save_dir, exist_ok = True)

    panoptic_evaluator = CocoPanopticMetric(
        ann_file = '/x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_anns_coco.json',
        seg_prefix = '/x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_seg_anns_coco',
        classwise = True
    )
    panoptic_evaluator.dataset_meta = metainfo
    detection_evaluator = CocoMetric(
        ann_file = '/x22201018/datasets/MedicalDatasets/PanNuke/Part3/detection_binary_anns_coco.json',
        metric = ['bbox', 'segm'],
        classwise = True
    )
    detection_evaluator.dataset_meta = metainfo

    # val
    model.load_parameters(args.ckpt_path)
    val_one_epoch_binary(model, dataloader, panoptic_evaluator, detection_evaluator)



'''
python scripts/prompt_seg/panoptic_eval.py \
    logs/prompt_seg/pannuke_binary/74_02 \
    logs/prompt_seg/pannuke_binary/74_02/checkpoints/best_miou.pth \
    --prompt_type random_bbox \
    --points_per_batch 128
'''
