import random
import numpy as np
import torch
from functools import partial
import numpy as np
import cv2
import copy

import multiprocessing

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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

def gene_max_area_box(image_mask_np):
    '''
    Args:
        image_mask_np(numpy): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        [centroid_x,centroid_y], [x1,y1,x2,y2]
    '''
    # 寻找连通域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=8)
    # 找到最大连通域的索引
    max_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # +1 to exclude background label
    
    # 获取最大连通域的包围盒坐标（左上角，右下角）
    x1,y1,w,h,_ = stats[max_area_index]
    x2,y2 = x1 + w, y1 + h
    # 获取最大连通域的中点位置（直接取中心位置有可能取在背景上：环状前景）
    [centroid_x,centroid_y] = centroids[max_area_index]
    centroid_x,centroid_y = int(centroid_x),int(centroid_y)
    if not image_mask_np[centroid_y, centroid_x]:
        # 中心点不在前景区域时，在最大连通区域内随机选一个点
        positive_coords = np.nonzero(labels == max_area_index)
        random_pos_index = np.random.choice(np.arange(positive_coords[0].shape[0]), size=1)
        centroid_y,centroid_x = positive_coords[0][random_pos_index],positive_coords[1][random_pos_index]
        centroid_y,centroid_x = centroid_y[0],centroid_x[0]
    return [centroid_x,centroid_y], [x1,y1,x2,y2]

def gene_bbox_for_mask(image_mask_np):
    '''
    Args:
        image_mask_np(numpy): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        [[x1,y1,x2,y2], ..., [x1,y1,x2,y2]]
    '''
    # 寻找连通域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=8)
    trans_box = lambda x1,y1,w,h: [x1,y1, x1 + w, y1 + h]
    # stats[0] 是背景框
    all_boxes = [trans_box(x1,y1,w,h) for x1,y1,w,h,_ in stats[1:]]
    return all_boxes

def get_prompt(prompt_type, binary_mask, boxes_candidate, device, coord_ratio=1):
    '''
    Notices:
        - binary_mask and boxes_candidate can only exist one of them.
        - if boxes_candidate is None and prompt type include box, the boxes_candidate will be
          generate from binary_mask.
        - the return values of boxes and points, there dim 0 means different
    
    Args:
        - prompt_type(string): prompt type, belong to 
            ['max_bbox', 'random_bbox', 'all_bboxes', 'random_point', 'max_bbox_center_point']
        - binary_mask(np.array): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
        - boxes_candidate(np.array): The candidate boxes. Expects value format is 
            [[x1,y1,x2,y2], ..., [x1,y1,x2,y2]].
        - device: tensor to device.
        - coord_ratio(int): box or point coords will be scaled to coord_ratio
    
    Returns:
        (boxes and point which meet requirements of SAM prompt encoder)
        - boxes(tensor): shape is (k, 1, 4), k is the num of boxes, 4 is (x1, y1, x2, y2)
        - point(tuple): (coords_torch, labels_torch), coords_torch shape is (bs, k, 2), 2 is (x,y)
            labels_torch shape is (bs, k), k is the num of points.
        - sampled_idx(tensor): the idx of sampled boxes or point.
    '''
    allowed_types = ['max_bbox', 'random_bbox', 'all_bboxes', 'random_point', 'max_bbox_center_point', 'max_bbox_with_point']
    assert prompt_type in allowed_types, \
            f'the prompt_type must be in {allowed_types}'
    assert binary_mask is not None or boxes_candidate is not None, \
            f'binary_mask and boxes_candidate can only exist one of them'
    
    
    boxes, point, sampled_idx = None, None, None

    if binary_mask is not None and np.sum(binary_mask) == 0:
        return boxes, point, sampled_idx
    if boxes_candidate is not None and len(boxes_candidate) == 0:
        return boxes, point, sampled_idx

    if prompt_type == 'random_point':
        input_points,_ = gene_points(binary_mask, 1, 0)
        random_point = input_points[0] * coord_ratio
        coords_torch = torch.as_tensor(np.array([random_point]), dtype=torch.float, device=device)
        labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        point = (coords_torch, labels_torch)
        return boxes, point, None
    
    if boxes_candidate is None:
        boxes_candidate = gene_bbox_for_mask(binary_mask)
        boxes_candidate = np.array(boxes_candidate)
    else:
        boxes_candidate = copy.deepcopy(boxes_candidate)
    
    boxes_candidate *= coord_ratio

    if prompt_type == 'random_bbox':
        random_bbox_idx = np.random.choice(range(len(boxes_candidate)), 1)[0]
        box_np = np.array(boxes_candidate[random_bbox_idx]) # [x1,y1,x2,y2]
        box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
        boxes = box_torch[None, :]  #[[[x1,y1,x2,y2]]]
        return boxes, point, [random_bbox_idx]
    if prompt_type == 'all_bboxes':
        box_np = np.array(boxes_candidate)
        boxes = torch.as_tensor(box_np, dtype=torch.float, device=device)
        boxes = boxes.unsqueeze(1)  # (k_box_nums, 1, 4)
        return boxes, point, np.arange(len(boxes_candidate))
   
    
    boxes_area = np.array([(x2-x1) * (y2-y1) for x1,y1,x2,y2 in boxes_candidate])
    max_idx = np.argmax(boxes_area)
    if prompt_type == 'max_bbox':
        box_np = np.array(boxes_candidate[max_idx]) # [x1,y1,x2,y2]
        box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
        boxes = box_torch[None, :]  #[[[x1,y1,x2,y2]]]
        return boxes, point, [max_idx]
    if prompt_type == 'max_bbox_center_point':
        x1,y1,x2,y2 = boxes_candidate[max_idx]
        centroid_x,centroid_y = int(x1+(x2-x1)/2),int(y1+(y2-y1)/2)
        coords_torch = torch.as_tensor(np.array([[centroid_x,centroid_y]]), dtype=torch.float, device=device)
        labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        point = (coords_torch, labels_torch)
        return boxes, point, [max_idx]
    if prompt_type == 'max_bbox_with_point':
        box_np = np.array(boxes_candidate[max_idx]) # [x1,y1,x2,y2]
        box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
        boxes = box_torch[None, :]  #[[[x1,y1,x2,y2]]]
    
        x1,y1,x2,y2 = boxes_candidate[max_idx]
        centroid_x,centroid_y = int(x1+(x2-x1)/2),int(y1+(y2-y1)/2)
        coords_torch = torch.as_tensor(np.array([[centroid_x,centroid_y]]), dtype=torch.float, device=device)
        labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        point = (coords_torch, labels_torch)
        return boxes, point, [max_idx]

def get_det(mask):
    '''
    mask: numpy.array, shape is (h,w)
    '''
    r = 4
    mask = mask.astype(int)
    height = mask.shape[0]  
    width  = mask.shape[1]
    det = np.zeros((height,width))
    for i in range(r, height-r):
        for j in range(r, width-r):
            y1,x1,y2,x2 = i-r, j-r, i+r, j+r
            if np.sum(mask[y1:y2, x1:x2]) < 0.0001:
                continue
            g1,g2,g3,g4,g5,g6,g7,g8 = 0,0,0,0,0,0,0,0
            rr = 2
            for k in range(rr):
                for l in range(k+1, r+1):
                    g1 += (mask[i,j+k] - mask[i,j+l])
                    g2 += (mask[i-k,j+k] - mask[i-l,j+l])
                    g3 += (mask[i-k,j] - mask[i-l,j])
                    g4 += (mask[i-k,j-k] - mask[i-l,j-l])
                    g5 += (mask[i,j-k] - mask[i,j-l])
                    g6 += (mask[i+k,j-k] - mask[i+l,j-l])
                    g7 += (mask[i+k,j] - mask[i+l,j])
                    g8 += (mask[i+k,j+k] - mask[i+l,j+l])
           
            if g1>0 and g2>0 and g3>0 and g4>0 and g5>0 and g6>0 and g7>0 and g8>0:
                det[i,j] = g1+g2+g3+g4+g5+g6+g7+g8
    
    if det.max() - det.min() != 0:
        det = (det-det.min())/(det.max()-det.min())
    return det


def split_matrix(matrix):
    h, w = matrix.shape
    return [matrix[:h//2, :w//2], matrix[:h//2, w//2:], matrix[h//2:, :w//2], matrix[h//2:, w//2:]]

def combine_matrices(submatrices):
    top = np.concatenate((submatrices[0], submatrices[1]), axis=1)
    bottom = np.concatenate((submatrices[2], submatrices[3]), axis=1)
    return np.concatenate((top, bottom), axis=0)

def get_det_multiprocessing(mask):
    mask = mask.astype(int)

    # 将矩阵分割成 4 个子矩阵
    submatrices = split_matrix(mask)
    # 创建进程池，切成多少块就用多少个核
    # cpu_num = multiprocessing.cpu_count()
    cpu_num = 4
    pool = multiprocessing.Pool(processes=cpu_num)

    # 异步处理每个子矩阵
    results = [pool.apply_async(get_det, (submatrix,)) for submatrix in submatrices]
    # 获取每个子矩阵的处理结果
    processed_submatrices = [result.get() for result in results]

    # 关闭进程池
    pool.close()
    pool.join()

    # 将处理后的子矩阵组合回一个完整的矩阵
    processed_matrix = combine_matrices(processed_submatrices)

    return processed_matrix

def get_connection(mask,thresh=100):
    '''
    input: n*n
    bbox: from 0
    idx: from 1
    '''
    queue = []
    regions = []
    minmaxs = []
    mask1 = mask.copy()
    flag = mask1.copy()
    output = np.zeros(mask1.shape)

    count = 0
    # thresh = 100

    #find 
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i,j]!=0 and flag[i,j]!=0:
                region = []
                minmax = np.zeros((mask1.shape[0],2))
                minmax[:,0] = mask1.shape[1]+1
                minmax[:,1] = -1
                queue.clear()
                region.append((i,j))
                queue.append((i,j))
                flag[i,j] = 0
                while len(queue)>0:
                    item = queue.pop(0)
                    if item[0]-1>=0 and mask1[item[0]-1,item[1]]!=0 and flag[item[0]-1,item[1]]!=0:
                        queue.append((item[0]-1,item[1]))
                        region.append((item[0]-1,item[1]))
                        flag[item[0]-1,item[1]] = 0
                        if minmax[item[0]-1,0]>item[1]:
                            minmax[item[0]-1,0] = item[1]
                        if minmax[item[0]-1,1]<item[1]:
                            minmax[item[0]-1,1] = item[1]
                    if item[0]+1<=mask1.shape[0]-1 and mask1[item[0]+1,item[1]]!=0 and flag[item[0]+1,item[1]]!=0:
                        queue.append((item[0]+1,item[1]))
                        region.append((item[0]+1,item[1]))
                        flag[item[0]+1,item[1]] = 0
                        if minmax[item[0]+1,0]>item[1]:
                            minmax[item[0]+1,0] = item[1]
                        if minmax[item[0]+1,1]<item[1]:
                            minmax[item[0]+1,1] = item[1]
                    if item[1]-1>=0 and mask1[item[0],item[1]-1]!=0 and flag[item[0],item[1]-1]!=0:
                        queue.append((item[0],item[1]-1))
                        region.append((item[0],item[1]-1))
                        flag[item[0],item[1]-1] = 0
                        if minmax[item[0],0]>item[1]-1:
                            minmax[item[0],0] = item[1]-1
                        if minmax[item[0],1]<item[1]-1:
                            minmax[item[0],1] = item[1]-1
                    if item[1]+1<=mask1.shape[1]-1 and mask1[item[0],item[1]+1]!=0 and flag[item[0],item[1]+1]!=0:
                        queue.append((item[0],item[1]+1))
                        region.append((item[0],item[1]+1))
                        flag[item[0],item[1]+1] = 0
                        if minmax[item[0],0]>item[1]+1:
                            minmax[item[0],0] = item[1]+1
                        if minmax[item[0],1]<item[1]+1:
                            minmax[item[0],1] = item[1]+1
                if len(region)<thresh:
                    continue
                
                a = list(region)
                count += 1
                b = minmax.copy()
                # print(id(region))
                # print(id(a))
                regions.append(a)
                minmaxs.append(b)
                for k in region:
                    output[k[0],k[1]] = count
    # bbox
    # [hmin,wmin,hmax,wmax]
    bboxs = []
    for minmax in minmaxs:
        hmin,wmin,hmax,wmax = 0,0,0,0
        inflag = 0
        for i in range(minmax.shape[0]):
            if minmax[i,0] > minmax[i,1]:
                if inflag == 1:
                    inflag = 0
                    break
                else:
                    continue
            if inflag == 0:
                inflag = 1
                hmin = i
                hmax = i
                wmin = minmax[i,0]
                wmax = minmax[i,1]
            else:
                hmax = i
                wmin = min(wmin,minmax[i,0])
                wmax = max(wmax,minmax[i,1])

        bboxs.append([int(hmin),int(wmin),int(hmax),int(wmax)])
    
    return output,bboxs

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

        valued_points, idx = np.unique(np.array(valued_points), return_index=True, axis=0)
        valued_bboxes = np.array(valued_bboxes)[idx]
        valued_points,valued_bboxes = valued_points.tolist(),valued_bboxes.tolist()

    return valued_points,edge_intensity,valued_bboxes
