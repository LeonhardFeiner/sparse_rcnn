import numpy as np
import torch
import torch.nn as nn
from ndsis.utils.bbox import (
    clip_boxes, bbox_transform, bbox_transform_inv, select_bbox,
    calc_start_end, anchor_overlap_size_distance, select_bbox2)


class AnchorStorage(nn.Module):
    def __init__(
            self,
            raw_anchor_levels,
            conv_stride_levels,
            store=True,
            store_goal_device=True,
            **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.store = store
        self.store_goal_device = store_goal_device
        self.descriptions = nn.ModuleDict()

        self.conv_stride_levels = nn.Parameter(
            torch.as_tensor(
                conv_stride_levels, dtype=torch.get_default_dtype()),
            requires_grad=False)

        self.raw_anchor_levels = nn.ParameterList(
            nn.Parameter(
                torch.as_tensor(raw_anchors, dtype=torch.get_default_dtype()),
                requires_grad=False)
            for raw_anchors in raw_anchor_levels)

    def __getitem__(self, key):
        innerkey = str(key)
        if innerkey in self.descriptions:
            anchor_description = self.descriptions[innerkey]
        else:
            scene_shape, conv_shape_levels = key
            anchor_description = AnchorDescriptionMultiLevel(
                scene_shape, conv_shape_levels, self.raw_anchor_levels,
                self.conv_stride_levels,
                store_goal_device=self.store_goal_device, **self.kwargs)

            if self.store:
                self.descriptions[innerkey] = anchor_description

        return anchor_description

    def __delitem__(self, key):
        self.descriptions.__delitem__(str(key))

    def __contains__(self, key):
        self.descriptions.__contains__(str(key))


class AnchorDescriptionMultiLevel(nn.Module):
    def __init__(
            self,
            scene_shape,
            conv_shape_levels,
            raw_anchor_levels,
            conv_stride_levels,
            allowed_border=0,
            store_goal_device=False,
            clip_boxes=True):
        super().__init__()
        dtype = conv_stride_levels.dtype
        device = conv_stride_levels.device

        store_device = device if store_goal_device else torch.device('cpu')

        self.clip_boxes = clip_boxes
        self.scene_shape = torch.as_tensor(
            scene_shape, dtype=dtype, device=device)
        self.num_dims = len(self.scene_shape)
        self.permute_inputs = (0,) + tuple(range(2, self.num_dims + 2)) + (1,)

        conv_shape_levels_dtype = torch.tensor(
            conv_shape_levels, dtype=dtype, device=device)

        anchors_levels = [
            AnchorDescriptionMultiLevel._get_pixelwise_anchors(*params)
            for params
            in zip(conv_shape_levels, conv_stride_levels, raw_anchor_levels)
        ]

        anchor_count_levels = [
            [len(raw_anchor_level)] for raw_anchor_level in raw_anchor_levels]
        self.rpn_score_shape = np.append(
            conv_shape_levels, anchor_count_levels, axis=1)
        self.anchors_shape = self.rpn_score_shape.copy()
        self.anchors_shape[:, -1] *= 2 * self.num_dims
        self.combined_shape = self.rpn_score_shape.copy()
        self.combined_shape[:, -1] *= 2 * self.num_dims + 1

        flat_anchors_levels = [
            anchors.reshape(-1, 2, self.num_dims)
            for anchors in anchors_levels]

        all_anchors = torch.cat(flat_anchors_levels, axis=0)

        lower_limit = -allowed_border
        upper_limit = self.scene_shape + allowed_border

        anchor_start, anchor_end = calc_start_end(*all_anchors.unbind(-2))
        self.inside_indicator = (
            torch.all(anchor_start >= lower_limit, axis=-1) &
            torch.all(anchor_end <= upper_limit, axis=-1)
        ).cpu()

        self.inside_anchors = (
            all_anchors[self.inside_indicator].to(store_device))

        self.num_anchor_levels = [
            len(anchors) for anchors in flat_anchors_levels]

        associated_levels = [
            torch.full((num_anchors,), level_index, dtype=torch.long)
            for level_index, num_anchors in enumerate(self.num_anchor_levels)
        ]

        self.inside_associated_level = (
            torch.cat(associated_levels, axis=0)[self.inside_indicator]
            .to(store_device))

    @staticmethod
    def _get_pixelwise_anchors(
            conv_shape, conv_stride_prod, raw_anchors, *, pixel_center=0.5):
        assert(len(conv_shape) == raw_anchors.shape[1])
        ranges = (
            (
                torch.arange(
                    shape_dim,
                    device=raw_anchors.device,
                    dtype=raw_anchors.dtype)
                + pixel_center
            ) * stride_prod_dim
            for shape_dim, stride_prod_dim
            in zip(conv_shape, conv_stride_prod))

        shifts = torch.cartesian_prod(*ranges).unsqueeze(-2)
        broadcasted_tensors = torch.broadcast_tensors(shifts, raw_anchors)
        stacked_tensors = torch.stack(broadcasted_tensors, axis=-2)

        return stacked_tensors

    def get_bbox_targets(self, gt_bbox_batch):
        inside_anchors = self.inside_anchors.to(gt_bbox_batch[0].device)

        (
            batch_max_overlaps_raw,
            batch_argmax_overlaps_raw,
            batch_gt_rois_raw
        ) = zip(*(
            select_bbox(inside_anchors, gt_boxes)
            for gt_boxes in gt_bbox_batch))

        batch_max_overlaps = torch.stack(batch_max_overlaps_raw, dim=0)
        batch_argmax_overlaps = torch.stack(batch_argmax_overlaps_raw, dim=0)
        batch_gt_rois = torch.stack(batch_gt_rois_raw, dim=0)

        batch_bbox_targets = bbox_transform(inside_anchors, batch_gt_rois)

        return batch_max_overlaps, batch_argmax_overlaps, batch_bbox_targets

    def rpn_permuter(self, x_levels):
        permuted_levels = [x.permute(self.permute_inputs) for x in x_levels]
        return permuted_levels

    def rpn_bbox_score_splitter(self, x_levels):
        shapes = np.array([permuted.shape for permuted in x_levels])
        assert((self.combined_shape == shapes[:, 1:]).all())

        single_anchor_dim = 2 * self.num_dims + 1

        concatenated = torch.cat([
            permuted.reshape(permuted.size(0), num_anchors, single_anchor_dim)
            for permuted, num_anchors
            in zip(x_levels, self.num_anchor_levels)], axis=1)

        inside = concatenated[:, self.inside_indicator]

        bbox_raw, score_raw = torch.split(
            inside, (2 * self.num_dims, 1), dim=-1)

        bbox = bbox_raw.reshape(
            *bbox_raw.shape[:-1], 2, self.num_dims)

        score = score_raw.squeeze(-1)

        return bbox, score

    def rpn_bbox_inside_selector(self, x_levels):
        permuted_levels = [x.permute(self.permute_inputs) for x in x_levels]
        shapes = np.array([permuted.shape for permuted in permuted_levels])
        assert((self.anchors_shape == shapes[:, 1:]).all())

        concatenated = torch.cat([
            permuted.reshape(permuted.size(0), num_anchors, 2, self.num_dims)
            for permuted, num_anchors
            in zip(permuted_levels, self.num_anchor_levels)], axis=1)

        return concatenated[:, self.inside_indicator]

    def rpn_score_inside_selector(self, x_levels):
        permuted_levels = [x.permute(self.permute_inputs) for x in x_levels]
        shapes = np.array([permuted.shape for permuted in permuted_levels])
        assert((self.rpn_score_shape == shapes[:, 1:]).all())

        concatenated = torch.cat([
            permuted.reshape(permuted.size(0), num_anchors)
            for permuted, num_anchors
            in zip(permuted_levels, self.num_anchor_levels)], axis=1)

        return concatenated[:, self.inside_indicator]

    def forward(self, rpn_bbox):
        inside_anchors = self.inside_anchors.to(rpn_bbox.device)
        raw_bbox = bbox_transform_inv(inside_anchors, rpn_bbox)

        proposal_bbox = (
            clip_boxes(raw_bbox, self.scene_shape)
            if self.clip_boxes else
            raw_bbox)

        return proposal_bbox

