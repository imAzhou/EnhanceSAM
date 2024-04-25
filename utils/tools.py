import random
import numpy as np
import torch
from functools import partial
import numpy as np
import cv2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    one_M = 1e6
    return {
        'Total': f'{(total_num/one_M):.4f}M',
        'Trainable': f'{(trainable_num/one_M):.4f}M',
    }

def one_hot_encoder(n_classes, input_tensor):
    '''
    Args:
        n_classes (int): number of classess
        input_tensor (Tensor): shape is (bs, h, w), 
            the pixel value belong to [0, n_classes-1]
    Return:
        output_tensor (Tensor): shape is (bs, n_classes, h, w), 
            the pixel value belong to [0, 1]
    '''
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def gene_points(image_mask, pos_sample_num, neg_sample_num):
    '''
    Args:
        image_mask(Tensor): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        input_point = np.array([[x, y], ..., [x, y]])
        input_label = np.array([1, ..., 0])
    '''
    input_point = []
    input_label = []
    # 获取前景点的所有坐标
    # positive_coords.shape: (total_positive_num, 2)
    positive_coords = np.nonzero(image_mask)
    if positive_coords.shape[0] > 0:
        # 有可能取的正样本点数量超过所有的前景像素点，replace=True : 可以重复选点
        random_pos_index = np.random.choice(np.arange(positive_coords.shape[0]), size=pos_sample_num,replace=True)
    else:
        random_pos_index = []
    # 获取背景点的所有坐标
    # negative_coords: tuple(np.array(Y),np.array(X))
    negative_coords = np.where(image_mask == 0)
    random_neg_index = np.random.choice(np.arange(len(negative_coords[0])), size=neg_sample_num,replace=True)
    
    # 返回坐标格式为 (x,y), image_mask的格式为 H*W，故其第一个值为 y，第二个值为x
    for idx in random_pos_index:
        input_point.append([positive_coords[idx][1].item(),positive_coords[idx][0].item()])
        input_label.append(1)
    
    for idx in random_neg_index:
        input_point.append([negative_coords[1][idx],negative_coords[0][idx]])
        input_label.append(0)

    return np.array(input_point),np.array(input_label)

def gene_point_embed(model,image_mask, point_num,device):
    points = None

    input_point, input_label = gene_points(image_mask,point_num[0],point_num[1])
    posi_point = np.nonzero(input_label)
    if posi_point[0].shape[0] > 0:
        # point_coords = transform.apply_coords(input_point, resize_size)
        coords_torch = torch.as_tensor(input_point, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        points = (coords_torch, labels_torch)

    sparse_embeddings, _ = model.prompt_encoder(
        points=points,
        boxes=None,
        masks=None,
    )
    return sparse_embeddings,points

def gene_max_area_center_point(image_mask_np):
    '''
    Args:
        image_mask_np(numpy): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        center_point_x, center_point_y
    '''
    # 寻找连通域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=8)
    # 找到最大连通域的索引
    max_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # +1 to exclude background label
    
    # 获取最大连通域的中点位置（直接取中心位置有可能取在背景上：环状前景）
    [centroid_x,centroid_y] = centroids[max_area_index]
    centroid_x,centroid_y = int(centroid_x),int(centroid_y)
    if not image_mask_np[centroid_y, centroid_x]:
        # 中心点不在前景区域时，在最大连通区域内随机选一个点
        positive_coords = np.nonzero(labels == max_area_index)
        random_pos_index = np.random.choice(np.arange(positive_coords[0].shape[0]), size=1)
        centroid_y,centroid_x = positive_coords[0][random_pos_index],positive_coords[1][random_pos_index]
        centroid_y,centroid_x = centroid_y[0],centroid_x[0]
    return centroid_x,centroid_y
