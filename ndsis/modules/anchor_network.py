import torch
import torch.nn as nn
from ndsis.modules.anchor import AnchorStorage
from ndsis.modules.custom_container import (
    ZipApply, ModuleMap, ConcatOutputListList)
import numpy as np
from itertools import chain


def get_anchor_network(
        conv_stride_levels,
        channels_levels,
        *,
        raw_anchor_levels,
        allowed_border=0,
        store_anchors=True,
        extra_stride_levels=None,
        store_goal_device=True):
    return (
        AnchorNetworkLinear(
            raw_anchor_levels, conv_stride_levels, channels_levels,
            allowed_border, store_anchors, store_goal_device)
        if extra_stride_levels is None else
        AnchorNetworkUpsample(
            raw_anchor_levels, conv_stride_levels, channels_levels,
            allowed_border, store_anchors, store_goal_device,
            extra_stride_levels=extra_stride_levels))


class AnchorNetworkLinear(nn.Module):
    def __init__(
            self, raw_anchor_levels, conv_stride_levels, channels_levels,
            allowed_border=0, store_anchors=True, store_goal_device=True):
        super().__init__()
        assert (
            len(raw_anchor_levels)
            == len(conv_stride_levels)
            == len(channels_levels))

        num_dims = raw_anchor_levels[0].shape[1]

        num_anchor_levels = [
            len(raw_anchors) for raw_anchors in raw_anchor_levels]

        self.rpn_net_levels = ZipApply(*[
             nn.Linear(rpn_channels, num_anchors * (num_dims * 2 + 1))
             for rpn_channels, num_anchors
             in zip(channels_levels, num_anchor_levels)])

        self.anchor_storage = AnchorStorage(
            raw_anchor_levels,
            store=store_anchors,
            store_goal_device=store_goal_device,
            conv_stride_levels=conv_stride_levels,
            allowed_border=allowed_border)

    def forward(self, rpn_feat_levels, scene_shape):
        # num_levels x [batch x conv_net_channels x spatial_dims]
        # conv_feat_levels = self.conv_net_levels(scene)

        feat_shapes = [
            tuple(rpn_feat.shape[2:]) for rpn_feat in rpn_feat_levels]
        anchor_description = self.anchor_storage[scene_shape, feat_shapes]
        permuted_feat_levels = anchor_description.rpn_permuter(rpn_feat_levels)

        rpn_net_out = self.rpn_net_levels(permuted_feat_levels)
        rpn_bbox, rpn_score = anchor_description.rpn_bbox_score_splitter(
            rpn_net_out)

        return rpn_bbox, rpn_score, anchor_description


class AnchorNetworkConv(nn.Module):
    def __init__(
            self, raw_anchor_levels, conv_stride_levels, channels_levels,
            allowed_border=0, store_anchors=True, store_goal_device=True):
        super().__init__()
        assert (
            len(raw_anchor_levels)
            == len(conv_stride_levels)
            == len(channels_levels))

        num_dims = raw_anchor_levels[0].shape[1]

        num_anchor_levels = [
            len(raw_anchors) for raw_anchors in raw_anchor_levels]

        self.rpn_net_levels = ZipApply(*[
            nn.Conv3d(
                rpn_channels, num_anchors * (num_dims * 2 + 1), [1, 1, 1])
            for rpn_channels, num_anchors
            in zip(channels_levels, num_anchor_levels)])

        self.anchor_storage = AnchorStorage(
            raw_anchor_levels,
            store=store_anchors,
            store_goal_device=store_goal_device,
            conv_stride_levels=conv_stride_levels,
            allowed_border=allowed_border)

    def forward(self, rpn_feat_levels, scene_shape):
        # num_levels x [batch x conv_net_channels x spatial_dims]
        # conv_feat_levels

        # num_levels x [batch x rpn_channels x spatial_dims
        # rpn_feat_levels

        feat_shapes = [tuple(rpn_feat.shape[2:]) for rpn_feat in rpn_feat_levels]
        anchor_description = self.anchor_storage[scene_shape, feat_shapes]

        # num_levels x [batch x spatial_dims x (num_anchors * 2 * num_dims)]
        rpn_output_raw = self.rpn_net_levels(rpn_feat_levels)


        # inside_anchors contains several dimensions
        #  (spatial_dims x num_anchors x num_levels)
        #  these dimensions are levelwise flattened and concatenated because
        #  every of these dimensions can be different for every geometric level
        #  furthermore the anchors which are not inside the image are removed

        permuted_output_raw = anchor_description.rpn_permuter(rpn_output_raw)
        rpn_bbox, rpn_score = anchor_description.rpn_bbox_score_splitter(permuted_output_raw)

        return rpn_bbox, rpn_score, anchor_description


class AnchorNetworkUpsample(nn.Module):
    @staticmethod
    def create_level_upsampler(
            conv_stride_level, channels_level,
            anchor_list_level, extra_strides_level):
        num_dims = 3

        assert all(
            (num_dims == raw_anchors.shape[1])
            for raw_anchors in anchor_list_level)

        num_anchors_level = [
            len(raw_anchors) for raw_anchors in anchor_list_level]

        anchor_layer = [
            nn.ConvTranspose3d(
                channels_level, num_anchors * (num_dims * 2 + 1),
                tuple(extra_stride), tuple(extra_stride))
            for num_anchors, extra_stride
            in zip(num_anchors_level, extra_strides_level)]

        combined_strides_level = conv_stride_level / extra_strides_level

        submodule = ModuleMap(*anchor_layer)

        return submodule, combined_strides_level

    def __init__(
            self, raw_anchor_levels, conv_stride_levels, channels_levels,
            allowed_border=0, store_anchors=True, store_goal_device=True, *,
            extra_stride_levels):
        super().__init__()
        assert(
            len(raw_anchor_levels)
            == len(conv_stride_levels)
            == len(channels_levels)
            == len(extra_stride_levels))
        assert all(
            (len(raw_anchors) == len(extra_strides))
            for raw_anchors, extra_strides
            in zip(raw_anchor_levels, extra_stride_levels))

        submodules, combined_strides_levels = zip(*(
            self.create_level_upsampler(
                conv_stride_level, channels_level,
                anchor_list_level, extra_strides_level)
            for (
                conv_stride_level, channels_level,
                anchor_list_level, extra_strides_level)
            in zip(
                conv_stride_levels, channels_levels,
                raw_anchor_levels, extra_stride_levels
            )))

        combined_strides = np.concatenate(combined_strides_levels)
        self.rpn_net_levels = ConcatOutputListList(ZipApply(*submodules))
        raw_anchor_levels = list(chain.from_iterable(raw_anchor_levels))

        self.anchor_storage = AnchorStorage(
            raw_anchor_levels,
            store=store_anchors,
            store_goal_device=store_goal_device,
            conv_stride_levels=combined_strides,
            allowed_border=allowed_border)


    def forward(self, rpn_feat_levels, scene_shape):
        # num_levels x [batch x conv_net_channels x spatial_dims]
        # conv_feat_levels

        # num_levels x [batch x rpn_channels x spatial_dims
        # rpn_feat_levels

        # num_levels x [batch x spatial_dims x (num_anchors * 2 * num_dims)]
        rpn_output_raw_levels = self.rpn_net_levels(rpn_feat_levels)

        feat_shapes = [
            tuple(rpn_output_raw.shape[2:])
            for rpn_output_raw in rpn_output_raw_levels]
        anchor_description = self.anchor_storage[scene_shape, feat_shapes]

        # inside_anchors contains several dimensions
        #  (spatial_dims x num_anchors x num_levels)
        #  these dimensions are levelwise flattened and concatenated because
        #  every of these dimensions can be different for every geometric level
        #  furthermore the anchors which are not inside the image are removed

        permuted_output_raw = anchor_description.rpn_permuter(
            rpn_output_raw_levels)
        rpn_bbox, rpn_score = anchor_description.rpn_bbox_score_splitter(
            permuted_output_raw)

        return rpn_bbox, rpn_score, anchor_description
