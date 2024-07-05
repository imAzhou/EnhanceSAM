import numpy as np
import torch
from torchvision.ops.boxes import batched_nms
from .tools import filter_boxes
from mmdet.evaluation.functional import INSTANCE_OFFSET
from typing import Tuple
from models.sam.utils.amg import (
    MaskData,
    batched_mask_to_box,
    calculate_stability_score,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

def process_batch(
        points: np.ndarray,     # (k_points, 2) in size scale 1024
        masks: torch.Tensor,    # (bs, 1, h, w) in size scale original image
        iou_preds: torch.Tensor,    # (bs, 1)
        cls_preds: torch.Tensor,    # (bs, 1)
        orig_size: Tuple[int, ...], # (256, 256)
        mask_threshold: float = 0.,
        pred_iou_thresh: float = 0.7,
        pred_cls_thresh: float = 0.5,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
    ) -> MaskData:
        orig_h, orig_w = orig_size
        crop_box = [0, 0, orig_w, orig_h]
        
        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            cls_preds=cls_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted cls (filter background mask)
        if pred_cls_thresh > 0.0:
            keep_mask = data["cls_preds"] > pred_cls_thresh
            data.filter(keep_mask)

        # Filter by predicted IoU
        if pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], mask_threshold, stability_score_offset
        )
        if stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        # del data["masks"]

        return data

def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

def process_batch_points(model, points, bs_image_embedding, device, filter=False, scale_ratio=4):

    coords_torch = torch.as_tensor(points, dtype=torch.float, device=device)
    labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
    coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
    point_prompt = (coords_torch, labels_torch)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=point_prompt,
        boxes=None,
        masks=None,
    )

    sam_outputs = model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = model.prompt_encoder.get_dense_pe(),
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
    cls_preds = torch.ones_like(sam_ious_process)

    pred_iou_thresh = 0.5 if filter else 0
    stability_score_thresh = 0.7 if filter else 0
    batch_data = process_batch(
        points // scale_ratio, sam_logits_process, sam_ious_process,cls_preds,
        orig_size = (256,256),
        pred_iou_thresh = pred_iou_thresh,
        stability_score_thresh = stability_score_thresh
    )

    return batch_data

def process_img_points(data:MaskData, ):
    h, w = 256,256
    min_mask_region_area = 32

    if len(data.items()) > 0:
        # Remove duplicates within this crop.
        box_nms_thresh = 0.7
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
        crop_box = [0, 0, h, w]
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        data.to_numpy()
        # Filter small disconnected regions and holes in masks
        if min_mask_region_area > 0:
            data = postprocess_small_regions(
                data,
                min_mask_region_area,
                box_nms_thresh,
            )
        data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]
    
    return data

def format2panoptic(data:MaskData, device):
    h, w = 256,256
    bg_clsid = 0
    cell_clsid = 1
    panoptic_seg = torch.full((h, w), bg_clsid, dtype=torch.int32, device=device)
    # ins_results = dict(
    #     bboxes = torch.as_tensor([]),
    #     labels = torch.as_tensor([]),
    #     scores = torch.as_tensor([]),
    #     masks = torch.as_tensor([]),
    # )

    if len(data.items()) > 0:
        bboxes = data['boxes']     # np.array, (k,2) size in original image
        masks = np.array(data["segmentations"])     # np.array, (k, h, w) size in original image

        instance_id = 1
        if len(bboxes) >0:
            for i in range(len(bboxes)):
                mask = masks[i]
                panoptic_seg[mask] = (cell_clsid + instance_id * INSTANCE_OFFSET)
                instance_id += 1
        # iou_preds = torch.as_tensor(data["iou_preds"])
        # ins_results = dict(
        #     bboxes = torch.as_tensor(bboxes),
        #     labels = torch.ones_like(iou_preds, dtype=torch.int16) * (cell_clsid-1),
        #     scores = iou_preds,
        #     masks = torch.as_tensor(masks)
        # )
    return panoptic_seg

def format2AJI(data_sample, pred_panoptic_seg: torch.Tensor):
    h,w = data_sample.img_shape
    gt_map = torch.full((h, w), 0, dtype=torch.int32)
    pred_map = torch.full((h, w), 0, dtype=torch.int32)

    gt_bitmap_mask = data_sample.gt_instances.masks.masks   #  np.array, shape is (k_instance, h, w)
    for instance_id,bitmap in enumerate(gt_bitmap_mask):
        gt_map[bitmap.astype(bool)] = instance_id+1
    
    pred_insid = torch.unique(pred_panoptic_seg)
    pred_insid = pred_insid[1:]     # remove background id
    for n_iid,o_iid in enumerate(pred_insid):
        pred_map[pred_panoptic_seg == o_iid] = n_iid+1

    return gt_map.numpy(),pred_map.numpy()