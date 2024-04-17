import torch

def sample_points_from_mask(coarse_mask, mask_index, num_points=50):
    """
    Randomly sample coordinates of non-zero points from a binary mask tensor.

    Args:
    - coarse_mask (torch.Tensor): Binary tensor with shape (bs, h, w)
    - mask_index (torch.Tensor): Integer tensor with shape (bs, h, w)
    - num_points (int): Number of points to be sampled.

    Returns:
    - sample_points (torch.Tensor): Tensor of sampled coordinates with shape (bs, num_points, 2), where each coordinate is in (y, x) format.
    - point_indices (torch.Tensor): Tensor of sampled points indices of all points sam mask index.
    """
    bs, h, w = coarse_mask.shape
    device = coarse_mask.device
    total_nonzero = (coarse_mask.view(bs, -1) != 0).sum(dim=1)
    sample_points = torch.ones((bs, num_points, 2), dtype=torch.long, device=device) * -1
    point_indices = torch.ones((bs, num_points), dtype=torch.long, device=device) * -1

    for i in range(bs):
        if total_nonzero[i] > 0:
            nonzero_indices = torch.nonzero(coarse_mask[i].view(-1) != 0, as_tuple=False)

            if total_nonzero[i] > num_points:
                selected_indices = torch.randperm(total_nonzero[i])[:num_points]            
            elif total_nonzero[i] > 0:
                selected_indices = torch.randint(0, total_nonzero[i], (num_points,))
            
            sample_points[i] = torch.stack([(nonzero_indices[selected_indices] // w).view(-1), (nonzero_indices[selected_indices] % w).view(-1)], dim=-1)
            # valid_points.shape: (50), incase points coords is -1
            valid_points = (sample_points[i] >= 0)[:, 0]
            # valid_points_coords.shape: (valid_nums, 2)
            valid_points_coords = sample_points[i][valid_points]
            point_indices[i][valid_points] = mask_index[i, valid_points_coords[:, 0], valid_points_coords[:, 1]]

    return sample_points, point_indices


def calculate_point_embed(image_embed, point_mask_indices, mask_index):
    """
    Extract values from another tensor based on valid indices and calculate the mean.

    Args:
    - image_embed (torch.Tensor): Tensor of embedded images with shape (bs, c, h, w)
    - point_mask_indices (torch.Tensor): Tensor of sampled points sam mask indices with shape (bs, num_points)
    - mask_index (torch.Tensor): Tensor of all points sam mask indices with shape (bs, h, w)
    - default_embedding (torch.Tensor): (1, c)

    Returns:
    - point_embed (torch.Tensor): Tensor of calculated point embeddings with shape (bs, num_points, c).
    """
    _,c,h,w = image_embed.shape
    device = image_embed.device
    bs, num_points = point_mask_indices.shape
    point_embed = torch.zeros((bs, num_points, c), dtype=image_embed.dtype, device=device)
    unique_index = torch.unique(point_mask_indices)
    for idx in unique_index:
        if idx >= 0: 
            # equal_indice.shape: (bs=16, h=256, w=256)
            equal_indice = (mask_index == idx).unsqueeze(1).repeat(1, c, 1, 1)
            
            # image_embed.shape: (bs=16, c=16, h=256, w=256)
            # zero_image_mask = torch.tensor(0).to(device)
            # image_embed_idx = torch.where(equal_indice, image_embed, zero_image_mask).mean(dim=(-1, -2))

            image_embed_idx = torch.masked_select(image_embed, equal_indice)
            # 统计每个样本中不为 0 的元素数量
            non_zero_counts = torch.sum(equal_indice, dim=(-1, -2))
            cumulative_sum = torch.cumsum(non_zero_counts.view(-1), dim=0)
            # 使用 torch.narrow 获取每个位置的值
            start_idx = torch.cat([torch.zeros(1, dtype=torch.int64).to(device), cumulative_sum[:-1]])
            values = [image_embed_idx.narrow(0, int(start), int(length)) for start, length in zip(start_idx, non_zero_counts.view(-1))]
            # 求和并计算平均值
            image_embed_idx = torch.stack([v.sum() / (h*w) for v in values]).view(bs, c)

            mask = point_mask_indices == idx
            replacement_values_expanded = image_embed_idx.unsqueeze(1).expand(-1, mask.size(1), -1)
            point_embed[mask] = replacement_values_expanded[mask]
    
    return point_embed
