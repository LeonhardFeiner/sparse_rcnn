"""Method collection for bounding box calculations.

This module provides methods for bounding box calculations.
This includes box clipping, box rounding, box transformations, box
    comparisions and non-maximum-suppression.
"""
from typing import Tuple
import torch


separator_dim = -2    # separates start from end or position from size
coordinates_dim = -1  # contains the dimensions of scene shape x, y, z, ...
boxes_dim = 0         # contains a list of boxes


def clip_boxes(
        boxes: torch.tensor,
        scene_shape: torch.tensor) -> torch.tensor:
    """Clip box coordinates to ensure them to be between 0 and scene_shape.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.

    Args:
        boxes (torch.tensor): :math:`(*, 2, N_{dim})`
            the start and stop coordinates of the boxes
        scene_shape (torch.tensor): :math:`(N_{dim},)`
            The shape of the scene or featuremap

    Returns:
        clipped_boxes: (torch.tensor): :math:`(*, 2, N_{dim})` the clipped
            start and stop coordinates of the boxes

    """
    return torch.min(boxes, scene_shape).clamp(min=0)


def clip_boxes_min_max(
        boxes: torch.tensor,
        min_coords: torch.tensor,
        max_coords: torch.tensor) -> torch.tensor:
    """Clip box coordinates to ensure them to be between min_coords and
        max_coords.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.

    Args:
        boxes (torch.tensor): :math:`(*, 2, N_{dim})`
            the start and stop coordinates of the boxes
        min_coords (torch.tensor): :math:`(N_{dim},)`
            the minimum allowed coordinate values
        max_coords (torch.tensor): :math:`(N_{dim},)`
            the maximum allowed coordinate values

    Returns:
        clipped_boxes: (torch.tensor): :math:`(*, 2, N_{dim})` the clipped
            start and stop coordinates of the boxes

    """
    return torch.max(torch.min(boxes, max_coords), min_coords)


def clip_boxes_asymmetric(
        boxes: torch.tensor,
        scene_shape: torch.tensor) -> torch.tensor:
    """Clip box coordinates to ensure the coordinates to be valid.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.
    Start coordinates are clipped between :math:`0` and :math:`scene_shape-1`.
    Stop coordinates are clipped between :math:`1` and :math:`scene_shape`.

    Args:
        boxes (torch.tensor): :math:`(*, 2, N_{dim})`
            the start and stop coordinates of the boxes
        scene_shape (torch.tensor): :math:`(N_{dim},)`
            The shape of the scene or featuremap

    Returns:
        clipped_boxes: (torch.tensor): :math:`(*, 2, N_{dim})` the clipped
            start and stop coordinates of the boxes

    """
    min_coords = boxes.new_tensor([[0], [1]])
    max_coords = torch.stack((scene_shape - 1, scene_shape))
    return clip_boxes_min_max(boxes, min_coords, max_coords)


def round_bbox(boxes):
    """Round the box coordinates in a way, that the box is bigger or equal in
        every dimension.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.
    Start coordinates are floored, stop coordinates are ceiled.
    Afterwards they are converted to long

    Args:
        boxes (torch.tensor): :math:`(*, 2, N_{dim})`
            the start and stop coordinates of the boxes

    Returns:
        rounded_boxes: (torch.LongTensor): :math:`(*, 2, N_{dim})`
            the rounded and typeconverted start and stop coordinates of the
            boxes

    """
    start, stop = boxes.unbind(-2)
    return torch.stack((start.floor(), stop.ceil()), dim=-2).long()


def bbox_transform_position_size(
        anchor_position: torch.tensor,
        anchor_size: torch.tensor,
        gt_position: torch.tensor,
        gt_size: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """Calculate regression targets for anchor refinement.

    All input Tensors must be broadcastable.

    Args:
        anchor_position (torch.tensor): :math:`(*, N_{dim})`
            the position of the center of the anchors
        anchor_size (torch.tensor): :math:`(*, N_{dim})`
            the size of the anchors
        gt_position (torch.tensor): :math:`(*, N_{dim})`
            the position of the center of the ground truth bounding boxes
        gt_size (torch.tensor): :math:`(*, N_{dim})`
            the size of the ground truth bounding boxes

    Returns:
        deltas_position (torch.tensor): :math:`(*, N_{dim})`
            the delta of position regression target for anchor refinement
        deltas_size (torch.tensor): :math:`(*, N_{dim})`
            the delta of size regression target for anchor refinement
    """
    deltas_position = ((gt_position - anchor_position) / (anchor_size + 1e-14))
    deltas_size = torch.log(gt_size / (anchor_size + 1e-14) + 1e-14)
    return deltas_position, deltas_size


def bbox_transform_inv_position_size(
        anchor_position: torch.tensor,
        anchor_size: torch.tensor,
        deltas_position: torch.tensor,
        deltas_size: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """Calculate the predicted bounding boxes.

    Args:
        anchor_position (torch.tensor): :math:`(*, N_{dim})`
            the position of the center of the anchors.
        anchor_size (torch.tensor): :math:`(*, N_{dim})`
            the size of the anchors
        deltas_position (torch.tensor): :math:`(*, N_{dim})`
            the delta of position network output for anchor refinement
        deltas_size (torch.tensor): :math:`(*, N_{dim})`
            the delta of size network output for anchor refinement

    Returns:
        gt_position (torch.tensor): :math:`(*, N_{dim})`
            the position of the center of the predicted bounding boxes
        gt_size (torch.tensor): :math:`(*, N_{dim})`
            the size of the predicted bounding boxes
    """
    pred_position = deltas_position * anchor_size + anchor_position
    pred_size = torch.exp(deltas_size) * anchor_size

    return pred_position, pred_size


def bbox_overlap_unsqueezed_size_start_end(
        boxes_a_size: torch.tensor,
        boxes_a_start: torch.tensor,
        boxes_a_end: torch.tensor,
        boxes_b_size: torch.tensor,
        boxes_b_start: torch.tensor,
        boxes_b_end: torch.tensor) -> torch.tensor:
    """Calculate the intersection over union of a list of boxes

    All tensor must be broadcastable to each other.

    Args:
        boxes_a_size (torch.tensor): :math:`(*, N_{dim})`
            the size of the first boxes
        boxes_a_start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the first boxes
        boxes_a_end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the first boxes
        boxes_b_size (torch.tensor): :math:`(*, N_{dim})`
            the size of the second boxes
        boxes_b_start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the second boxes
        boxes_b_end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the second boxes

    Returns:
        overlaps (torch.tensor): :math:`(*)`
            the intersections over union between boxes of both tensors against
            the other tensor
    """
    boxes_a_area = boxes_a_size.prod(coordinates_dim)
    boxes_b_area = boxes_b_size.prod(coordinates_dim)
    return bbox_overlap_unsqueezed_area_start_end(
        boxes_a_area, boxes_a_start, boxes_a_end,
        boxes_b_area, boxes_b_start, boxes_b_end)


def bbox_overlap_unsqueezed_area_start_end(
        boxes_a_area: torch.tensor,
        boxes_a_start: torch.tensor,
        boxes_a_end: torch.tensor,
        boxes_b_area: torch.tensor,
        boxes_b_start: torch.tensor,
        boxes_b_end: torch.tensor) -> torch.tensor:
    """Calculate the intersection over union of a list of boxes

    All tensor must be broadcastable to each other.

    Args:
        boxes_a_area (torch.tensor): :math:`(*, N_{dim})`
            the area or volume of the first boxes
        boxes_a_start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the first boxes
        boxes_a_end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the first boxes
        boxes_b_area (torch.tensor): :math:`(*, N_{dim})`
            the area or volume of the second boxes
        boxes_b_start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the second boxes
        boxes_b_end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the second boxes

    Returns:
        overlaps (torch.tensor): :math:`(*)`
            the intersections over union between boxes of both tensors against
            the other tensor
    """
    max_start = torch.max(boxes_a_start, boxes_b_start)
    min_end = torch.min(boxes_a_end, boxes_b_end)
    intersection_area = (
        (min_end - max_start).clamp(min=0).prod(coordinates_dim))
    union_area = boxes_a_area + boxes_b_area - intersection_area
    overlaps = intersection_area / union_area

    return overlaps


def calc_position_size(
        start: torch.tensor,
        end: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """Converts start and stop coordinates of boxes to their size and position.

    Args:
        start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the boxes
        end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the boxes

    Returns:
        position (torch.tensor): :math:`(*, N_{dim})`
            the position of the center of the boxes
        size (torch.tensor): :math:`(*, N_{dim})`
            the size of the boxes
    """
    size = end - start
    position = start + 0.5 * size
    return position, size


def calc_start_end(
        position: torch.tensor,
        size: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """Converts size and position of boxes to their start and stop coordinates.

    Args:
        position (torch.tensor): :math:`(*, N_{dim})`
            the position of the center of the boxes
        size (torch.tensor): :math:`(*, N_{dim})`
            the size of the boxes

    Returns:
        start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the boxes
        end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the boxes
    """
    half_size = size / 2
    start = position - half_size
    end = position + half_size
    return start, end


def prepare_overlap_from_start_end(
        box: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Calculates size, start and end of boxes from start and stop coordinates.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.

    Args:
        box (torch.tensor): :math:`(*, 2, N_{dim})`
            the start and stop coordinates of the boxes

    Returns:
        size (torch.tensor): :math:`(*, N_{dim})`
            the size of the boxes
        start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the boxes
        end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the boxes
    """
    start, end = box.unbind(separator_dim)
    size = end - start
    return size, start, end


def prepare_overlap_from_position_size(
        box: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Calculates size, start and end of boxes the position of the center and
        the size of the boxes.

    Position of center and size have to be stacked at dimension :math:`-2`.

    Args:
        box (torch.tensor): :math:`(*, 2, N_{dim})`
            the position of the center and the size of the boxes

    Returns:
        size (torch.tensor): :math:`(*, N_{dim})`
            the size of the boxes
        start (torch.tensor): :math:`(*, N_{dim})`
            the start coordinates of the boxes
        end (torch.tensor): :math:`(*, N_{dim})`
            the stop coordinates of the boxes
    """
    position, size = box.unbind(separator_dim)
    start, end = calc_start_end(position, size)
    return size, start, end


def bbox_transform(
        anchors: torch.tensor,
        gt_bbox: torch.tensor) -> torch.tensor:
    """Calculate regression targets for anchor refinement.

    All input Tensors must be broadcastable.
    Start and stop coordinates of ground truth boxes, position of center and
        size of anchors have to be stacked at dimension :math:`-2`.

    Args:
        anchors (torch.tensor): :math:`(*, 2, N_{dim})`
            the position of the center and the size of the anchors
        gt_bbox (torch.tensor): :math:`(*, 2, N_{dim})`
            the start and the stop coordinates of the ground truth boxes

    Returns:
        deltas (torch.tensor): :math:`(*, 2, N_{dim})`
            the delta of position of the center and the size regression target
            for anchor refinement

    """
    deltas_position, deltas_size = bbox_transform_position_size(
        *anchors.unbind(separator_dim),
        *calc_position_size(*gt_bbox.unbind(separator_dim)))

    deltas = torch.stack((deltas_position, deltas_size), dim=separator_dim)

    return deltas


def bbox_transform_inv(
        anchors: torch.tensor,
        deltas: torch.tensor) -> torch.tensor:
    """Calculate the predicted bounding boxes.

    All input Tensors must be broadcastable.
    Start and stop coordinates of ground truth boxes, the delta of position of
        the center and the size have to be stacked at dimension :math:`-2`.

    Args:
        anchors (torch.tensor): :math:`(*, 2, N_{dim})`
            the position of the center and the size of the anchors
        deltas (torch.tensor): :math:`(*, 2, N_{dim})`
            the delta of position and the delta of size network output for
            anchor refinement

    Returns:
        gt_position (torch.tensor): :math:`(*, 2, N_{dim})`
            the position of the center and the size of the predicted bounding
            boxes

    """
    # anchors = batch_fit(anchors, deltas)

    pred_position, pred_size = bbox_transform_inv_position_size(
        *anchors.unbind(separator_dim),
        *deltas.unbind(separator_dim))

    pred_bbox = torch.stack(
        calc_start_end(pred_position, pred_size),
        dim=separator_dim)

    return pred_bbox


def bbox_overlap_anchors(
        anchors: torch.tensor,
        gt_bbox: torch.tensor) -> torch.tensor:
    """Calculate the intersection over union of anchors and ground truth boxes.

    Start and stop coordinates of ground truth boxes, position of center and
        size of anchors have to be stacked at dimension :math:`-2`.

    Args:
        anchors (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the position of the centers and the size of the anchors
        gt_bbox (torch.tensor): :math:`(N_{boxes}, *, 2, N_{dim})`
            the start and stop coordinate of the ground truth boxes

    Returns:
        overlaps (torch.tensor): :math:`(N_{anchors}, N_{boxes}, *)`
            the intersections over union between anchors and ground truth boxes

    """
    size_a, start_a, end_a = prepare_overlap_from_position_size(
        anchors.unsqueeze(boxes_dim + 1))

    size_b, start_b, end_b = prepare_overlap_from_start_end(
        gt_bbox.unsqueeze(boxes_dim))

    return bbox_overlap_unsqueezed_size_start_end(
        size_a, start_a, end_a, size_b, start_b, end_b)


def bbox_overlap_anchors2(
        anchors: torch.tensor,
        gt_bbox: torch.tensor) -> torch.tensor:
    """Calculate the intersection over union of anchors and ground truth boxes.

    Start and stop coordinates of ground truth boxes and anchors have to be
        stacked at dimension :math:`-2`.

    Args:
        anchors (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the position of the centers and the size of the anchors
        gt_bbox (torch.tensor): :math:`(N_{boxes}, *, 2, N_{dim})`
            the position of the centers and the size of ground truth boxes

    Returns:
        overlaps (torch.tensor): :math:`(N_{anchors}, N_{boxes}, *)`
            the intersections over union between anchors and ground truth boxes

    """
    size_a, start_a, end_a = prepare_overlap_from_position_size(
        anchors.unsqueeze(boxes_dim + 1))

    size_b, start_b, end_b = prepare_overlap_from_position_size(
        gt_bbox.unsqueeze(boxes_dim))

    return bbox_overlap_unsqueezed_size_start_end(
        size_a, start_a, end_a, size_b, start_b, end_b)


def bbox_overlap_anchors_split_combine(
        anchors: torch.tensor,
        gt_bbox: torch.tensor,
        max_anchors=2**18) -> torch.tensor:
    """Calculate the intersection over union of anchors and ground truth boxes.

    Start and stop coordinates of ground truth boxes, position of center and
        size of anchors have to be stacked at dimension :math:`-2`.
    The anchors are splitted into subarrays during calculation to reduce
        required gpu memory

    Args:
        anchors (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the position of the centers and the size of the anchors
        gt_bbox (torch.tensor): :math:`(N_{boxes}, *, 2, N_{dim})`
            the start and stop coordinate of the ground truth boxes
        max_anchors (int): max number of anchors in partial tensors

    Returns:
        overlaps (torch.tensor): :math:`(N_{anchors}, N_{boxes}, *)`
            the intersections over union between anchors and ground truth boxes

    """
    size_b, start_b, end_b = prepare_overlap_from_start_end(
        gt_bbox.unsqueeze(boxes_dim))

    anchor_list = torch.split(anchors, max_anchors)

    size_start_end_a = (
        prepare_overlap_from_position_size(
            single_anchors.unsqueeze(boxes_dim + 1))
        for single_anchors in anchor_list)

    overlaps = [
        bbox_overlap_unsqueezed_size_start_end(
            size_a, start_a, end_a, size_b, start_b, end_b)
        for size_a, start_a, end_a in size_start_end_a]

    return torch.cat(overlaps, axis=boxes_dim)


def bbox_overlap_anchors_split_combine2(
        anchors: torch.tensor,
        gt_bbox: torch.tensor,
        max_anchors=2**18) -> torch.tensor:
    """Calculate the intersection over union of anchors and ground truth boxes.

    Start and stop coordinates of ground truth boxes and anchors have to be
        stacked at dimension :math:`-2`.
    The anchors are splitted into subarrays during calculation to reduce
        required gpu memory

    Args:
        anchors (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the position of the centers and the size of the anchors
        gt_bbox (torch.tensor): :math:`(N_{boxes}, *, 2, N_{dim})`
            the position of the centers and the size of ground truth boxes
        max_anchors (int): max number of anchors in partial tensors

    Returns:
        overlaps (torch.tensor): :math:`(N_{anchors}, N_{boxes}, *)`
            the intersections over union between anchors and ground truth boxes

    """
    size_b, start_b, end_b = prepare_overlap_from_position_size(
        gt_bbox.unsqueeze(boxes_dim))

    anchor_list = torch.split(anchors, max_anchors)

    size_start_end_a = (
        prepare_overlap_from_position_size(
            single_anchors.unsqueeze(boxes_dim + 1))
        for single_anchors in anchor_list)

    overlaps = [
        bbox_overlap_unsqueezed_size_start_end(
            size_a, start_a, end_a, size_b, start_b, end_b)
        for size_a, start_a, end_a in size_start_end_a]

    return torch.cat(overlaps, axis=boxes_dim)


def bbox_overlap_prediction(
        roi_bbox: torch.tensor,
        gt_bbox: torch.tensor) -> torch.tensor:
    """Calculate the intersection over union of anchors and ground truth boxes.

    Start and stop coordinates of ground truth boxes and anchors have to be
        stacked at dimension :math:`-2`.
    The anchors are splitted into subarrays during calculation to reduce
        required gpu memory

    Args:
        roi_bbox (torch.tensor): :math:`(N_{rois}, *, 2, N_{dim})`
            the position of the centers and the size of the anchors
        gt_bbox (torch.tensor): :math:`(N_{boxes}, *, 2, N_{dim})`
            the position of the centers and the size of ground truth boxes

    Returns:
        overlaps (torch.tensor): :math:`(N_{rois}, N_{boxes}, *)`
            the intersections over union between region of interest boxes and
                ground truth boxes

    """
    size_a, start_a, end_a = prepare_overlap_from_start_end(
        roi_bbox.unsqueeze(boxes_dim + 1))

    size_b, start_b, end_b = prepare_overlap_from_start_end(
        gt_bbox.unsqueeze(boxes_dim))

    return bbox_overlap_unsqueezed_size_start_end(
        size_a, start_a, end_a, size_b, start_b, end_b)


def anchor_overlap_size_distance(
        anchor_size: torch.tensor,
        anchor_distances: torch.tensor) -> torch.tensor:
    """Calculates the overlap of the same anchors when shifted.

    Args:
        anchor_size (torch.tensor): :math:`(*, N_{dim})`
            the size of the anchors
        anchor_distances (torch.tensor): :math:`(*, N_{dim})`
            the distance between the position of the centers

    Returns:
        overlaps (torch.tensor): :math:`(*)`
            the intersections over union between the original and the shifted
            anchors
    """
    start_a = torch.zeros_like(anchor_distances)
    start_b = anchor_distances
    end_a = anchor_size
    end_b = anchor_size + anchor_distances
    return bbox_overlap_unsqueezed_size_start_end(
        anchor_size, start_a, end_a, anchor_size, start_b, end_b)


def bbox_self_overlap(boxes):
    """Calculates the overlap of the same anchors when shifted.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.

    Args:
        boxes (torch.tensor): :math:`(*, N_{boxes}, 2, N_{dim})`
            the start and stop positions of the boxes

    Returns:
        overlaps (torch.tensor): :math:`(*, N_{boxes}, N_{boxes})`
            the intersections over union between the boxes to each other
    """
    size, start, end = prepare_overlap_from_start_end(boxes)
    area = size.prod(coordinates_dim)
    area_a = area.unsqueeze(-3 + 1)
    start_a = start.unsqueeze(-3)
    end_a = end.unsqueeze(-3)
    area_b = area.unsqueeze(-2 + 1)
    start_b = start.unsqueeze(-2)
    end_b = end.unsqueeze(-2)
    return bbox_overlap_unsqueezed_area_start_end(
        area_a, start_a, end_a, area_b, start_b, end_b)


def select_bbox(
        anchors: torch.tensor,
        gt_bbox: torch.tensor) -> Tuple[
            torch.tensor, torch.tensor, torch.tensor]:
    """Selects gt bounding boxes for every anchor with highest overlap.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.

    Args:
        anchors (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the position of the centers and the size of the anchors
        gt_bbox (torch.tensor): :math:`(N_{boxes}, *, 2, N_{dim})`
            the position of the centers and the size of ground truth boxes

    Returns:
        max_overlaps (torch.tensor): :math:`(N_{anchors}, *)`
            the maximum overlap over all ground truth boxes
            only zeros if no ground truth boxes exist
        argmax_overlaps (torch.LongTensor): :math:`(N_{boxes}, *)`
            the index of the maximum overlap over all ground truth boxes
            only zeros if no ground truth boxes exist
        gt_bbox_associated (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the ground truth boxes which are associated with the anchor
            only zeros if no ground truth boxes exist
    """
    if len(gt_bbox) > 0:
        overlaps = bbox_overlap_anchors_split_combine(anchors, gt_bbox)
        max_overlaps, argmax_overlaps = overlaps.max(1)
        gt_bbox_associated = gt_bbox[argmax_overlaps]
    else:
        max_overlaps = torch.zeros_like(anchors[:, 0, 0])
        gt_bbox_associated = torch.zeros_like(anchors)
        argmax_overlaps = torch.full_like(max_overlaps, -1, dtype=torch.long)
    return max_overlaps, argmax_overlaps, gt_bbox_associated


def select_bbox2(
        anchors: torch.tensor,
        gt_bbox: torch.tensor,
        min_gt_bbox_size: float,
        return_clamped: bool = False) -> Tuple[
            torch.tensor, torch.tensor, torch.tensor]:
    """Selects gt bounding boxes for every anchor with highest overlap.

    Increases the size of small boxes up to `min_gt_bbox_size` to make
    overlaps of small boxes higher. if `returned_clamped` is `True`, also
    return the boxes with increased size instead of the originals.
    Start and stop coordinates have to be stacked at dimension :math:`-2`.

    Args:
        anchors (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the position of the centers and the size of the anchors
        gt_bbox (torch.tensor): :math:`(N_{boxes}, *, 2, N_{dim})`
            the position of the centers and the size of ground truth boxes

    Returns:
        max_overlaps (torch.tensor): :math:`(N_{anchors}, *)`
            the maximum overlap over all ground truth boxes
            only zeros if no ground truth boxes exist
        argmax_overlaps (torch.LongTensor): :math:`(N_{boxes}, *)`
            the index of the maximum overlap over all ground truth boxes
            only zeros if no ground truth boxes exist
        gt_bbox_associated (torch.tensor): :math:`(N_{anchors}, *, 2, N_{dim})`
            the ground truth boxes which are associated with the anchor
            only zeros if no ground truth boxes exist
        min_gt_bbox_size (float): the minimum size
        return_clamped (bool, optional) defaults to `False`
            True if also return the modified ground truth boxes should be
            returned instead of the originals.
    """
    if len(gt_bbox) > 0:
        gt_bbox_position, gt_bbox_size = calc_position_size(
            *gt_bbox.unbind(separator_dim))
        gt_bbox_size_clamped = gt_bbox_size.clamp(min_gt_bbox_size)
        gt_bbox_clamped = torch.stack(calc_start_end(
            gt_bbox_position, gt_bbox_size_clamped), axis=separator_dim)

        overlaps = bbox_overlap_anchors_split_combine(
            anchors, gt_bbox_clamped)
        max_overlaps, argmax_overlaps = overlaps.max(1)

        gt_bbox_associated = (
            gt_bbox_clamped if return_clamped else gt_bbox)[argmax_overlaps]
    else:
        max_overlaps = torch.zeros_like(anchors[:, 0, 0])
        gt_bbox_associated = torch.zeros_like(anchors)
        argmax_overlaps = torch.full_like(max_overlaps, -1, dtype=torch.long)
    return max_overlaps, argmax_overlaps, gt_bbox_associated


def non_maximum_supression(
        proposed_boxes: torch.tensor,
        overlap_threshold: float) -> torch.tensor:
    """Calculate the overlap of the same anchors when shifted.

    Start and stop coordinates have to be stacked at dimension :math:`-2`.
    This version of non maximum supression is vectorized over the batch.
    This comes with the downside that even already suppressed boxes must be
    calculated.
    Probably only faster on GPU, but also works on CPU in principle
    No cuda code is required for this method as there are no nested loops.

    Args:
        proposed_boxes (torch.tensor): :math:`(*, N_{boxes}, 2, N_{dim})`
            the start and stop positions of the proposed boxes
        overlap_threshold (float): the threshold of overlap which decides if
            boxes should be deleted because they probably belong to the same
            ground truth box

    Returns:
        passed_indicator (torch.BoolTensor): :math:`(*, N_{boxes})`
            Indicates whether a box should be kept
    """
    # get a overlap matrix over all high confidence boxes whithin a scene.
    overlap = bbox_self_overlap(proposed_boxes.detach())

    # check if this overlap is bigger than a threshold
    # store only overlaps of boxes with higher confidence against lowers
    # only higher confidence boxes should only be able to deactivate lower
    # confidence boxes but neither themselves nor higher confidence boxes
    treshold_matrix = torch.tril((overlap > overlap_threshold), diagonal=-1)

    # Tensor elements indicate whether box should be kept
    is_maximum = treshold_matrix.new_ones(proposed_boxes.shape[:-2])

    # loop over all boxes with highest confidence in the scene
    # Apply this vectorized over all boxes in the batch.
    for box in treshold_matrix.unbind(-1):

        # Disable all other boxes in the same scene if the current box is not
        # disabled.
        is_maximum &= ~box

        # Also disable the overlaps of boxes which getting disabled right now.
        treshold_matrix &= ~box.unsqueeze(-2)

    return is_maximum
