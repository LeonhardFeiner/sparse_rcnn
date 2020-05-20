# -*- coding: utf-8 -*-
"""Method collection for comparing, extracting and inserting masks.

This module provides methods for dense and sparse mask calculations.
Inserting and extracting masks into and from tensors is  only provided dense.
Methods for comparing masks can be used for both dense and sparse mask tensors.
Comparing methods include methods for pairwise and matrix comparisions.
"""
from typing import Tuple, List
import torch
import torch.nn as nn


def mask_iou_inner(mask_a, mask_b, sum_dims, dtype):
    """Calculate the intersection over union of two lists of masks.

    All tensor must be broadcastable to each other.

    Args:
        mask_a (torch.BoolTensor): :math:`(*, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of a mask of the first mask list
        mask_b (torch.BoolTensor): :math:`(*, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of a mask of the second mask list
        sum_dims (Tuple[int]):
            the indices of the dimensions containing a single mask
            these dimensions are removed within the final calculation
        dtype:
            the datatype used for calculation of the overlap values

    Returns:
        overlaps (torch.tensor): :math:`(*)`
            the intersections over union between masks of both tensors against
            the other tensor
    """
    inter = (mask_a & mask_b).type(dtype).sum(sum_dims)
    union = (mask_a | mask_b).type(dtype).sum(sum_dims)
    return inter / union


def mask_iou_pair(mask_a, mask_b, dtype=torch.get_default_dtype()):
    """Calculate the intersection over union of two masks.

    Args:
        mask_a (torch.BoolTensor): :math:`(*mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of the first mask
        mask_b (torch.BoolTensor): :math:`(*mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of the second mask
        dtype: Defaults to torch.float
            the datatype used for calculation of the overlap values

    Returns:
        overlaps (torch.tensor): :math:`()`
            the intersection over union between both masks
    """
    return mask_iou_inner(mask_a, mask_b, (), dtype)


def mask_confusion_pair(pred, gt, sum_dims=()):
    """Calculate the intersection over union of two masks.

    Args:
        pred (torch.BoolTensor): :math:`(*, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of the first mask list
        gt (torch.BoolTensor): :math:`(*, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of the second mask list
        sum_dims (Tuple[int]): Defaults to tuple()
            the indices of the dimensions containing a single mask
            these dimensions are removed within the final calculation
            empty tuple means all input dimensions are removed

    Returns:
        iou_matrix (torch.tensor): :math:`(*, 2,2)`
            the intersection over union between both masks
    """
    if sum_dims == ():
        sum_dims = tuple(range(gt.ndim))
    else:
        sum_dims = tuple(dim if dim > 0 else dim - 2 for dim in sum_dims)
    iou_matrix = (
        torch.stack((gt, ~gt), -1).unsqueeze(-2) &
        torch.stack((pred, ~pred), -1).unsqueeze(-1)).sum(sum_dims)
    return iou_matrix


def mask_iou_matrix(mask_a, mask_b, dtype=torch.get_default_dtype()):
    """Calculate the intersection over union of two lists of masks.

    Args:
        mask_a (torch.BoolTensor): :math:`(N_a, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of a mask of the first mask list
        mask_b (torch.BoolTensor): :math:`(N_b, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of a mask of the second mask list
        dtype: Defaults to torch.float
            the datatype used for calculation of the overlap values

    Returns:
        overlaps (torch.tensor): :math:`(N_a, N_b)`
            the intersections over union between masks of both tensors against
            the other tensor
    """
    unsqueezed_a = mask_a.unsqueeze(1)
    unsqueezed_b = mask_b.unsqueeze(0)
    sum_dims = tuple(range(2, unsqueezed_a.ndim))
    return mask_iou_inner(unsqueezed_a, unsqueezed_b, sum_dims, dtype)


def mask_iou_matrix_split_combine(
        mask_a: torch.tensor,
        mask_b: torch.tensor,
        max_masks_a=32,
        dtype=torch.get_default_dtype()) -> torch.tensor:
    """Calculate the intersection over union of two lists of masks.

    Works with reduced memory requirements.

    Args:
        mask_a (torch.BoolTensor): :math:`(N_a, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of a mask of the first mask list
        mask_b (torch.BoolTensor): :math:`(N_b, *mask_shape)`
            a tensor of indicator, whether each item (pixel, voxel, point)
            is part of a mask of the second mask list
        max_masks_a (int):
            mask_a are splitted into chunks of  size max_masks_a to reduce
            memory requirements during computation.
        dtype: Defaults to torch.float
            the datatype used for calculation of the overlap values

    Returns:
        overlaps (torch.tensor): :math:`(N_a, N_b)`
            the intersections over union between masks of both tensors against
            the other tensor
    """
    mask_a_list = torch.split(mask_a, max_masks_a, dim=0)

    overlaps = [
        mask_iou_matrix(single_mask_a, mask_b, dtype=dtype)
        for single_mask_a in mask_a_list]

    return torch.cat(overlaps, dim=0)


def get_slices_tuple(
        single_bbox: torch.LongTensor) -> Tuple[slice]:
    """Generate a slice tuple out of a bounding box tensor.

    Args:
        single_box (torch.LongTensor): :math:`(2, N_{dim})`
            the start and stop coordinates of the box

    Returns:
        slice_tuple: (Tuple[slice]):
            the slices created by the start and stop coordinates for dimension
    """
    return tuple(
        slice(start, stop) for start, stop in single_bbox.unbind(-1))


def select_mask(
        featuremap: torch.tensor,
        bbox: torch.LongTensor) -> torch.tensor:
    """Extract a rectangular bounding box region out of a tensor.

    This method works for dense n-dimensional data.
    *spatial_dims are 2 values for images or 3 values for voxel-grids
    *selected_spatial_dims are 2 values for images or 3 values for
        voxel-grids. The values are equal to the bounding box shape.
    Args:
        featuremap (torch.tensor): :math:`(*, *spatial_dims)`
            the full size featuremap which should be used to extract
        bbox (torch.LongTensor): :math:`(2, N_{dim})`
            the start and stop coordinates of the bounding box

    Returns:
        selected_features (torch.tensor):
            :math:`(*, *selected_spatial_dims)`:
            the selected features according to the bbox
    """
    return featuremap[(Ellipsis,) + get_slices_tuple(bbox)]


def sample_select_mask(
        featuremap: torch.tensor,
        bbox_list: torch.LongTensor) -> List[torch.tensor]:
    """Extract a rectangular bounding box region out of a tensor.

    This method works for dense n-dimensional data.
    *spatial_dims are 2 values for images or 3 values for voxel-grids
    *selected_spatial_dims are 2 values for images or 3 values for
        voxel-grids. The values are equal to the bounding box shape.
    Args:
        featuremap (torch.tensor): :math:`(*, *spatial_dims)`
            the full size featuremap which should be used to extract
        bbox_list (torch.LongTensor): :math:`(N_{box}, 2, N_{dim})`
            the start and stop coordinates of `N_{box}` bounding boxes

    Returns:
        selected_features (List[torch.tensor]):
            :math:`N_{box} x (*, *selected_spatial_dims)`:
            A list of selected feature maps of different shapes according
            to the bbox_list
    """
    return [
        featuremap[(Ellipsis,) + get_slices_tuple(bbox)] for bbox in bbox_list]


def insert_mask(
        mask: torch.tensor,
        bbox: torch.LongTensor,
        scene_size) -> torch.tensor:
    """Insert a mask into a tensor of full scene shape.

    Insert the features of a rectangular bounding box region into a full
        size tensor. The values out of the regions are padded with zeros.
    This method works for dense n-dimensional data.
    *spatial_dims are 2 values for images or 3 values for voxel-grids
    *selected_spatial_dims are 2 values for images or 3 values for
        voxel-grids. The values are equal to the bounding box shape.
    Args:
        mask (torch.tensor): :math:`(*, *selected_spatial_dims)`
            the selected region tensors which should be inserted
        bbox (torch.LongTensor): :math:`(2, N_{dim})`
            the start and stop coordinates of the bounding box

    Returns:
        full_size_mask: List of :math:`(*, *spatial_dims)`:
            the full size masks with inserted features of the region
    """
    full_mask = mask.new_zeros(scene_size)
    full_mask[get_slices_tuple(bbox)] = mask
    return full_mask


def insert_mask_list(
        mask_list: List[torch.tensor],
        bbox: torch.LongTensor,
        scene_size,
        default_type: torch.dtype = torch.bool,
        default_device: torch.device = torch.device('cpu')) -> torch.tensor:
    """Insert masks into a tensor consisting of a batch of full scene shape.

    Insert the features of rectangular bounding box regions into a full
        size tensor. The values out of the regions are padded with zeros.
    This method works for dense n-dimensional data.
    *spatial_dims are 2 values for images or 3 values for voxel-grids
    *selected_spatial_dims are 2 values for images or 3 values for
        voxel-grids. The values are equal to the bounding box shape.
    Args:
        mask_list (List[torch.tensor]):
            :math:`N_{box} x (*, *selected_spatial_dims)`
            the selected region tensors which should be inserted
        bbox (torch.LongTensor): :math:`(N_{box}, 2, N_{dim})`
            the start and stop coordinates of `N_{box}` bounding boxes

    Returns:
        full_size_masks: List of :math:`(N_{box}, *, *spatial_dims)`:
            the full size masks with inserted features of the region
    """
    if len(bbox):
        return torch.stack([
            insert_mask(single_mask, single_bbox, scene_size)
            for single_mask, single_bbox
            in zip(mask_list, bbox)
        ])
    else:
        return torch.zeros(
            (0, *scene_size), device=default_device, dtype=default_type)
