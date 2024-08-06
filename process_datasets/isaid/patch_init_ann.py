#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 02:40:23 2019

@author: felllix
@modified by akshitac8

"""
from pycocotools.coco import COCO
from prettytable import PrettyTable
import cv2
import os
import numpy as np
from pycocotools import mask as maskUtils
import tifffile
import mmcv
import json
from tqdm.contrib import tzip


data_root = '/x22201018/datasets/RemoteSensingDatasets/iSAID'
modes1 = ['train','val']
BACKGROUND_ID = 1
FOREGROUND_ID = 2
iSAID_palette = \
    {
        0: [0, 0, 0],
        1: [0, 0, 63],
        2: [0, 63, 63],
        3: [0, 63, 0],
        4: [0, 63, 127],
        5: [0, 63, 191],
        6: [0, 63, 255],
        7: [0, 127, 63],
        8: [0, 127, 127],
        9: [0, 0, 127],
        10: [0, 0, 191],
        11: [0, 0, 255],
        12: [0, 191, 127],
        13: [0, 127, 191],
        14: [0, 127, 255],
        15: [0, 100, 155]
    }

def statistic_cat_imgs():
    '''
    [{'id': 1, 'name': 'storage_tank'}, {'id': 2, 'name': 'Large_Vehicle'}, {'id': 3, 'name': 'Small_Vehicle'}, {'id': 4, 'name': 'plane'}, {'id': 5, 'name': 'ship'}, {'id': 6, 'name': 'Swimming_pool'}, {'id': 7, 'name': 'Harbor'}, {'id': 8, 'name': 'tennis_court'}, {'id': 9, 'name': 'Ground_Track_Field'}, {'id': 10, 'name': 'Soccer_ball_field'}, {'id': 11, 'name': 'baseball_diamond'}, {'id': 12, 'name': 'Bridge'}, {'id': 13, 'name': 'basketball_court'}, {'id': 14, 'name': 'Roundabout'}, {'id': 15, 'name': 'Helicopter'}]
    '''

    for mode in modes1:
        annotation_file = f'{data_root}/{mode}/Annotations/iSAID_{mode}.json'
        coco = COCO(annotation_file)
        # 获取所有的类别ID
        cat_ids = coco.getCatIds()
        # 获取所有的类别信息
        cats = coco.loadCats(cat_ids)
        table_data = PrettyTable()
        cat_names,cat_img_nums, cat_inst_nums = [],[],[]
        for cat_info in cats:
            load_cat_ids = coco.getCatIds(catNms=[cat_info['name']])
            img_ids = coco.getImgIds(catIds=load_cat_ids)
            ann_ids = coco.getAnnIds(catIds=load_cat_ids)
            cat_names.append(cat_info['name'])
            cat_img_nums.append(len(img_ids))
            cat_inst_nums.append(len(ann_ids))
        table_data.add_column('categories',cat_names)
        table_data.add_column('image nums',cat_img_nums)
        table_data.add_column('instance nums',cat_inst_nums)
        print(f'mode: {mode}\n' + table_data.get_string())

def split_and_filter(img, anns):
    img_H, img_W, _ = img.shape
    if img_H < patch_H and img_W > patch_W:
        img = mmcv.impad(img, shape=(patch_H, img_W), pad_val=0)
        img_H, img_W, _ = img.shape
    elif img_H > patch_H and img_W < patch_W:
        img = mmcv.impad(img, shape=(img_H, patch_W), pad_val=0)
        img_H, img_W, _ = img.shape
    elif img_H < patch_H and img_W < patch_W:
        img = mmcv.impad(img, shape=(patch_H, patch_W), pad_val=0)
        img_H, img_W, _ = img.shape

    # all pixes default background
    panoptic_ann = np.zeros((img_H, img_W, 3), dtype=np.int16)
    panoptic_ann[:, :, 0] = BACKGROUND_ID
    iid = 1
    for ann in anns:
        if 'segmentation' in ann:
            segmentation = ann['segmentation']
            if type(segmentation) == list:  # polygon
                for seg in segmentation:
                    poly = np.array(seg).reshape((len(seg)//2, 2)).astype(np.int32)
                    cv2.fillPoly(panoptic_ann, [poly], color=(FOREGROUND_ID, iid, 0))
            else:  # mask
                rle = maskUtils.frPyObjects(segmentation, img_H, img_W)
                mask = maskUtils.decode(rle)
                mask = np.array(mask, dtype=np.uint8)
                panoptic_ann[mask == 1] = (FOREGROUND_ID, iid, 0)  # 将mask区域绘制为绿色
            iid += 1

    img_patches,inst_mask_patches = [],[]
    for x in range(0, img_W, patch_W-overlap):
        for y in range(0, img_H, patch_H-overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str-=diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str-=diff_y
                y_end = img_H
            
            img_patch = img[y_str:y_end,x_str:x_end,:]
            inst_mask_patch = panoptic_ann[y_str:y_end,x_str:x_end,:]
            iid_set = np.unique(inst_mask_patch[:,:,1])
            foreground_area = np.sum(inst_mask_patch[:,:,0] == FOREGROUND_ID)
            foreground_ratio = foreground_area / (patch_W*patch_H)
            if len(iid_set) > 1 and foreground_ratio>=0.05:
                img_patches.append(img_patch)
                inst_mask_patches.append(inst_mask_patch)
    return img_patches,inst_mask_patches


def remap_inst_iid(inst_mask):
    inst_iids = np.unique(inst_mask)
    new_iid = 1
    for old_iid in inst_iids[1:]:
        mask = inst_mask == old_iid
        inst_mask[mask] = new_iid
        new_iid += 1
    return inst_mask

def patch_cat_imgs_labels():
    
    cat_id,cat_name,cat_tag = load_cat['id'],load_cat['name'],load_cat['save_tag']
    palette=[[255, 255, 255], iSAID_palette[cat_id]]

    categories_info = [
        {'id': BACKGROUND_ID, 'name': 'Background', 'isthing': 0, 'color':palette[0], 'supercategory': '' },
        {'id': FOREGROUND_ID, 'name': cat_name, 'isthing': 1, 'color':palette[1], 'supercategory': '' }
    ]

    for mode in modes1:
        src_img_dir = f'{data_root}/{mode}/images'
        
        target_dir = f'{data_root}/{mode}_{cat_tag}'
        target_img_dir = f'{target_dir}/img_dir'
        target_ann_mask_dir = f'{target_dir}/panoptic_seg_anns'
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_ann_mask_dir, exist_ok=True)

        init_ann_json = {
            'categories': categories_info
        }
        all_imgs_info = []
        current_img_id = 0

        annotation_file = f'{data_root}/{mode}/Annotations/iSAID_{mode}.json'
        coco = COCO(annotation_file)
        load_cat_ids = coco.getCatIds(catNms=[cat_name])
        img_ids = coco.getImgIds(catIds=load_cat_ids)
        images = coco.loadImgs(img_ids)

        files = [img_info['file_name'].split('.')[0] for img_info in images]
        for file_, img_id in tzip(files, img_ids):
            img_path = f'{src_img_dir}/{file_}.png'
            img = cv2.imread(img_path)
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=load_cat_ids)
            anns = coco.loadAnns(ann_ids)

            img_patches,inst_mask_patches = split_and_filter(img, anns)
            for img_p,inst_mask_p,idx in zip(img_patches,inst_mask_patches,range(len(img_patches))):
                save_img_path = f'{target_img_dir}/{file_}_{idx}.png'
                cv2.imwrite(save_img_path,img_p)
                inst_mask_p = remap_inst_iid(inst_mask_p)
                save_inst_mask_path = f'{target_ann_mask_dir}/{file_}_{idx}.tif'
                tifffile.imwrite(save_inst_mask_path, inst_mask_p)
                all_imgs_info.append({
                    'file_name': f'{file_}_{idx}.png',
                    'height': patch_H,
                    'width': patch_W,
                    'id': current_img_id,
                })
                current_img_id += 1
        
        init_ann_json['images'] = all_imgs_info
        json_save_path = f'{target_dir}/panoptic_anns.json'
        with open(json_save_path, 'w', encoding='utf-8') as ann_f:
            json.dump(init_ann_json, ann_f)
            

if __name__ == '__main__':
    # statistic_cat_imgs()
    patch_H, patch_W = 896, 896 # image patch width and height
    overlap = 384 #overlap area
    load_cat = {
        'id': 2, 'name': 'Large_Vehicle', 'save_tag': 'LV'
    }
    patch_cat_imgs_labels()
