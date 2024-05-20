import numpy as np
import matplotlib.pyplot as plt
import torch

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

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

def draw_pred(sampled_batch, pred_mask, pred_save_dir, coords_torch, box):
    '''
    Args:
        pred_mask: tensor, shape is (h,w), h=w=1024
        points: tensor, (num_points, 2), 2 is (x,y)
        box: tensor, [x1,y1,x2,y2]
    
    '''
    image_name = sampled_batch['meta_info']['img_name'][0]
    img = sampled_batch['input_image'][0].permute(1,2,0).numpy()
    gt = sampled_batch['mask_1024'][0]

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
