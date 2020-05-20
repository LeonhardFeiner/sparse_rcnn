import torch
import torch.nn as nn
from ndsis.utils.basic_functions import product
from ndsis.utils.mask import select_mask
from ndsis.modules.roi_select_bbox_transform import (
    BBoxTransformerSlice, BBoxTransformerInterpolation)


class RoiCut(nn.Module):
    def __init__(self, clip_boxes=True, resize_boxes=None):
        super().__init__()
        self.bbox_transformer = BBoxTransformerSlice(
            clip=clip_boxes, resize=resize_boxes)

    def forward(self, feature_map, bbox_batch):
        bbox_tensor, bbox_sample_count, bbox_sample_association = (
            self.bbox_transformer(bbox_batch, feature_map.shape[2:]))

        box_features = [
            select_mask(feature_map[sample_association], single_bbox)
            for sample_association, single_bbox
            in zip(bbox_sample_association, bbox_tensor)]

        scene_shape = feature_map.shape[2:]
        return box_features, (bbox_tensor, bbox_sample_count, scene_shape)


class RoiAlign(nn.Module):
    def __init__(
            self, extract_shape, padding=None,
            clip_boxes=False, resize_boxes=None):
        super().__init__()
        self.bbox_transformer = BBoxTransformerInterpolation(
            clip=clip_boxes, resize=resize_boxes)
        self.roi_align_inner = RoiAlignInner(extract_shape, padding)

    def forward(self, featuremap, bbox_batch):
        bbox_tensor, bbox_sample_count, bbox_sample_association = (
            self.bbox_transformer(bbox_batch, featuremap.shape[2:]))

        box_features = self.roi_align_inner(
            featuremap, bbox_tensor, bbox_sample_association)

        scene_shape = featuremap.shape[2:]
        return box_features, (bbox_tensor, bbox_sample_count, scene_shape)

    @staticmethod
    def split(feature_map, split_sections):
        return torch.split(feature_map, split_sections)

    @staticmethod
    def select(feature_map, indices):
        return feature_map[indices]


class RoiAlignInner(nn.Module):
    def __init__(self, extract_shape, padding=None):
        super(RoiAlignInner, self).__init__()

        ndim = len(extract_shape)
        self.extract_shape = extract_shape
        self.views = [
            (-1,) +
            tuple(2 if i == j else 1 for i in range(ndim)) +
            tuple(new_shape_dim if i == j else 1 for i in range(ndim))
            for j, new_shape_dim in enumerate(extract_shape)]
        self.batch_view = (-1,) + ((1,) * (2 * ndim))
        self.sum_dimensions = tuple(range(1, ndim + 1))
        self.permution_order = 0, ndim + 1, *range(1, ndim + 1)
        self.padding = padding

    def forward(self, featuremap, boxes, box_sample_association):
        box_shape = (len(box_sample_association), 2, len(self.extract_shape))
        feature_ndim = len(self.extract_shape) + 2
        assert boxes.shape == box_shape, (boxes.shape, box_shape)
        assert featuremap.ndim == feature_ndim, (featuremap.ndim, feature_ndim)
        assert box_sample_association.ndim == 1, box_sample_association.ndim

        device_dtype = dict(device=featuremap.device, dtype=featuremap.dtype)

        start, stop = boxes.unsqueeze(-3).unbind(-2)
        shape_minus_one_tensor = boxes.new_tensor(self.extract_shape) - 1
        step_width = ((stop - start) / shape_minus_one_tensor)

        coords_continuous_raw = (
            torch.arange(sample_count, **device_dtype)
            * sample_step_width + sample_start
            for sample_step_width, sample_start, sample_count
            in zip(
                step_width.unbind(-1), start.unbind(-1), self.extract_shape))

        coords_continuous = [
            torch.min(coords_continuous_tensor_raw, stop)
            for coords_continuous_tensor_raw, stop
            in zip(coords_continuous_raw, stop.unbind(-1))]

        coords_floored = [
            coords_continuous_tensor.floor()
            for coords_continuous_tensor in coords_continuous]
        coords_ceiled = [
            coords_continuous_tensor.ceil()
            for coords_continuous_tensor in coords_continuous]

        coords_raw = (
            torch.stack(rounded_tuple, axis=1).long()
            for rounded_tuple
            in zip(coords_floored, coords_ceiled))

        coords = [
            coords_dim.view(*view_dim)
            for coords_dim, view_dim
            in zip(coords_raw, self.views)]

        second_weights = (
            continuous - floored
            for continuous, floored
            in zip(coords_continuous, coords_floored))

        weights_raw = [
            torch.stack((1 - second_weight, second_weight), axis=1)
            for second_weight in second_weights]

        weights = [
            weights_dim.view(*view_dim, 1)
            for weights_dim, view_dim
            in zip(weights_raw, self.views)]

        featuremap_padded = (
            self.pad(coords, featuremap)
            if self.padding is not None else
            featuremap)

        batch_coords = box_sample_association.view(*self.batch_view)
        combined_coords = (batch_coords, slice(None), *coords)

        extracted = featuremap_padded[combined_coords]
        weighted_extract = extracted * product(weights)
        new_coords_raw = weighted_extract.sum(self.sum_dimensions)
        new_coords = new_coords_raw.permute(*self.permution_order)

        return new_coords

    def pad(self, coords, featuremap):
        min_tensor = torch.stack([c.min() for c in coords])
        max_tensor = torch.stack([c.max() for c in coords])
        feature_shape_tensor = torch.tensor(
            featuremap.shape, device=max_tensor.device)

        add_num_pixels = torch.max(
            -min_tensor, max_tensor - feature_shape_tensor).clamp(min=0)

        total_add_count = torch.stack(
            (torch.zeros_like(add_num_pixels), add_num_pixels), -1)

        padding_tuple = tuple(total_add_count.flatten())

        return torch.nn.functional.pad(
            featuremap, padding_tuple, mode='constant', value=self.padding)
