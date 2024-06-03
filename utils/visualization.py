import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

def show_mask(mask, ax, random_color=False, rgb=[30,144,255]):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([rgb[0]/255, rgb[1]/255, rgb[2]/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_multi_mask(mask_multi_cls, ax, palette):
    for cls_i,rgb in enumerate(palette):
        mask = mask_multi_cls == cls_i
        show_mask(mask, ax, rgb=rgb)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, edgecolor='green', min=0, max=1023):
    box = np.clip(np.array(box), min, max) 
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def draw_pred(sampled_batch, bs_idx, pred_mask, pred_save_dir, coords_torch=None, box=None):
    '''
    Args:
        pred_mask: tensor, shape is (h,w), h=w=1024
        points: tensor, (num_points, 2), 2 is (x,y)
        box: tensor, [x1,y1,x2,y2]
    '''
    image_name = sampled_batch['meta_info']['img_name'][bs_idx]
    img = sampled_batch['input_image'][bs_idx].permute(1,2,0).numpy()
    gt = sampled_batch['mask_1024'][bs_idx]

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    ax.imshow(img)
    show_mask(gt.cpu(), ax)
    ax.set_title('gt mask')

    ax = fig.add_subplot(122)
    ax.imshow(img)
    show_mask(pred_mask.cpu(), ax)
    if coords_torch is not None:
        labels_torch = torch.ones(len(coords_torch))
        show_points(coords_torch, labels_torch, ax)
    if box is not None:
        show_box(box, ax)
    ax.set_title('pred mask')

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()


def draw_multi_cls_pred(sampled_batch, bs_idx, pred_mask, pred_save_dir, metainfo, coords_torch, boxes, all_cls_agnostic_boxes):
    '''
    Args:
        pred_mask: tensor, shape is (h,w), h=w=1024, pred_mask value belong to [0, num_cls - 1]
        points: list, [num_cls, 2], 2 is (x,y)
        box: list, [num_cls, 4], 4 is [x1,y1,x2,y2]
        all_cls_agnostic_boxes: list, [k_boxes, 4], 4 is [x1,y1,x2,y2]
    '''
    image_name = sampled_batch['meta_info']['img_name'][bs_idx]
    img = sampled_batch['input_image'][bs_idx].permute(1,2,0).numpy()
    gt = sampled_batch['mask_1024'][bs_idx]

    classes = metainfo['classes']
    palette = metainfo['palette']

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    ax.imshow(img)
    show_multi_mask(gt.cpu(), ax, palette)
    ax.set_title('gt mask')

    ax = fig.add_subplot(122)
    ax.imshow(img)
    show_multi_mask(pred_mask.cpu(), ax, palette)
    if coords_torch is not None:
        labels_torch = torch.ones(len(coords_torch))
        show_points(coords_torch, labels_torch, ax)
    
    for i,box in enumerate(boxes):
        if box is not None:
            edgecolor = np.array([palette[i][0]/255, palette[i][1]/255, palette[i][2]/255, 1])
            show_box(box, ax, edgecolor=edgecolor)
    
    if all_cls_agnostic_boxes is not None:
        for box in all_cls_agnostic_boxes:
            show_box(box, ax)
    ax.set_title('pred mask')

    patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(len(classes))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def drwa_credible_region(image_name, img, gt, sam_seg, credible_p, credible_n, coords_torch, pred_save_dir):
    '''
    绘制四宫格，从上至下从左至右分别是：
        整图GT，选点及sam分割的块，
        可信前景区域，可信背景区域
    Args:
        img: np array, shape is (h,w,c)
        gt, sam_seg, credible_p, credible_n: tensor, shape is (h,w), value is 0/1
        coords_torch: (num_points, 2), 2 is (x,y)
    '''
    labels_torch = torch.ones(len(coords_torch))

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(221)
    ax.imshow(img)
    show_mask(gt.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('pred mask')

    ax = fig.add_subplot(222)
    ax.imshow(img)
    show_mask(sam_seg.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('SAM seg')

    ax = fig.add_subplot(223)
    ax.imshow(img)
    show_mask(credible_p.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('credible positive region')

    ax = fig.add_subplot(224)
    ax.imshow(img)
    show_mask(credible_n.cpu(), ax)
    show_points(coords_torch, labels_torch, ax)
    ax.set_title('credible negative region')

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()
