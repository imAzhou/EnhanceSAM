import matplotlib.pyplot as plt
import numpy as np
import cv2
from panopticapi.utils import rgb2id
from scipy.ndimage import binary_dilation, binary_erosion

# 提取边界函数
def extract_boundary(mask):
    dilated_mask = binary_dilation(mask)
    eroded_mask = binary_erosion(mask)
    boundary = dilated_mask ^ eroded_mask
    return boundary

def extract_inst_boundary(inst_mask):
    iid = np.unique(inst_mask)
    h,w = inst_mask.shape
    inst_boundary_mask = np.zeros((len(iid)-1, h, w))
    for i,id in enumerate(iid[1:]):
        iid_mask = (inst_mask == id).astype(np.float32)
        boundary_mask = extract_boundary(iid_mask)
        inst_boundary_mask[i] = boundary_mask
    inst_boundary_mask = np.max(inst_boundary_mask, axis=0)
    return inst_boundary_mask

def gtmap2inst(gt_mask):
    h,w,_ = gt_mask.shape
    result = np.zeros((h,w))
    id_mask = rgb2id(gt_mask)
    zero_mask = (gt_mask == 255)[:,:,0]
    bg_id = id_mask[zero_mask][0]
    iid = np.unique(id_mask)
    for i, id in enumerate(iid):
        if id != bg_id:
            result[id_mask == id] = i+1
    
    return result

def gene_insts_color(img, pan_result, boundary):
    h,w = 256,256
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(pan_result)) - 1
    
    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(131)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title('gt mask')

    ax = fig.add_subplot(132)
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[pan_result==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_axis_off()
    ax.set_title('color gt')
    
    ax = fig.add_subplot(133)
    ax.imshow(boundary, cmap='gray')
    ax.set_axis_off()
    ax.set_title('gt boundary')

    plt.tight_layout()
    plt.savefig('colorful_mask.png')
    plt.close()


img = cv2.imread('/x22201018/datasets/MedicalDatasets/MoNuSeg/train_p256_o64/panoptic_seg_anns_coco/TCGA-18-5592-01Z-00-DX1_0.png')
pan_result = gtmap2inst(img)
# 提取边界
boundary_mask = extract_inst_boundary(pan_result)
gene_insts_color(img, pan_result, boundary_mask)

