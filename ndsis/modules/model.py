# %%
import torch.nn as nn
import torch
import numpy as np
from collections import namedtuple
import sparseconvnet as scn
from ndsis.modules.custom_container import (
    SequentialInterims, ZipApply, Elementwise)
from ndsis.modules.anchor_network import get_anchor_network
from ndsis.modules.proposal_selector import get_roi_selector
from ndsis.modules.custom_operations import CustomInputLayer
from ndsis.utils.bbox import bbox_overlap_prediction as bbox_overlap
from ndsis.utils.mask import insert_mask, select_mask
from ndsis.utils.basic_functions import (
    default_argmax, split_select_nd, split_list)
from ndsis.modules.roi_select_dense import RoiAlign, RoiCut as DenseRoiCut
from ndsis.modules.roi_select_sparse import (
    RawToRawFeatureExtractorCombiner, RawToTensorFeatureExtractorCombiner,
    TensorToTensorFeatureExtractorCombiner,
    RawToFeaturesSceneFeatureExtractorCombiner, SparseRoiCut,
    SparseRoiExtraCut)
from ndsis.modules.module_factory import *


FeatureLevelDescriptor = namedtuple(
    'FeatureLevelDescriptor',
    ['type', 'channels', 'params', 'anchor_path'],
    defaults=(None, dict(), None))


class InstanceSegmentationNetwork(nn.Module):
    def __init__(
            self,
            *,
            feature_extractor_params,
            anchor_network_params,
            roi_selector_params,
            class_network_params=dict(),
            mask_network_params=dict(),
            segmentation_params=dict(),
            include_bbox_network=True,
            include_class_network=False,
            include_mask_network=False,
            include_segmentation_network=False
            ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(**feature_extractor_params)

        self.default_roi_tensor = nn.Parameter(
            torch.zeros(
                (0, 2, self.feature_extractor.num_dims),
                dtype=torch.get_default_dtype()),
            requires_grad=False)

        if include_bbox_network:
            self.bbox_network = get_anchor_network(
                self.feature_extractor.anchor_strides,
                self.feature_extractor.anchor_channels,
                **anchor_network_params)
            self.roi_selector = get_roi_selector(**roi_selector_params)
        else:
            self.bbox_network = None
            self.roi_selector = None

        if include_class_network:
            self.class_network = ClassNetwork(
                self.feature_extractor.num_dims,
                self.feature_extractor.class_sparse,
                self.feature_extractor.class_channels,
                self.feature_extractor.class_stride,
                **class_network_params)
            self.class_predictor = ClassPredictor()
        else:
            self.class_network = None
            self.class_predictor = None

        if include_mask_network:
            self.mask_network = SparseMaskNetwork(
                    self.feature_extractor.num_dims,
                    self.feature_extractor.sparse,
                    self.feature_extractor.input_channels,
                    self.feature_extractor.skip_connection_channels,
                    self.feature_extractor.skip_connection_strides,
                    self.feature_extractor.unet_channels,
                    self.feature_extractor.unet_strides,
                    **mask_network_params)
            self.mask_predictor = (
                SparseMaskPredictor
                if self.feature_extractor.sparse else
                DenseMaskPredictor)(self.mask_network.classes)
        else:
            self.mask_network = None
            self.mask_predictor = None

        self.overlap_calculator = OverlapCalculator()

        if include_segmentation_network:
            self.segmentation_network = SegmentationNetwork(
                    self.feature_extractor.num_dims,
                    self.feature_extractor.sparse,
                    self.feature_extractor.unet_strides,
                    self.feature_extractor.unet_channels,
                    **segmentation_params)
            self.segmentation_predictor = SegmentationPredictor(
                self.segmentation_network.sparse)
        else:
            self.segmentation_network = None
            self.segmentation_predictor = None

        self.calc_bbox = include_bbox_network
        self.calc_classes = include_class_network
        self.calc_masks = include_mask_network
        self.calc_segmentation = include_segmentation_network

    def forward(
            self, scene,
            calc_bbox=False, calc_classes=False, calc_masks=False,
            calc_segmentation=False, gt_bbox=None, gt_label=None,
            calc_gt_predictions=False, calc_mask_predictions=True):

        calc_segmentation |= self.calc_segmentation
        calc_classes |= self.calc_classes
        calc_masks |= self.calc_masks
        calc_bbox |= self.calc_bbox
        # spatial_dims can be arbitrary number of dimensions (non-flattened)
        #  eg. num_dims = 3:  spatial_dims = w x h x l
        # they can be different for every level

        # [batch x spatial_dims x features]
        (
            scene_size, batch_size, box_feature_map_levels, class_feature_map,
            mask_feature_map, unet_feature_maps
        ) = self.feature_extractor(scene)

        return_dict = dict(is_training=self.training)

        if calc_segmentation:
            segmentation = self.segmentation_network(unet_feature_maps, scene)
            return_dict['spn_segmentation'] = segmentation
        if not self.training and calc_segmentation:
            (
                return_dict['segmentation_class'],
                return_dict['segmentation_probabilites']
            ) = self.segmentation_predictor(segmentation)

        if calc_bbox:
            rpn_bbox, rpn_score, anchor_description = self.bbox_network(
                box_feature_map_levels, scene_size)

            return_dict = dict(
                return_dict,
                rpn_bbox=rpn_bbox,
                rpn_score=rpn_score,
                rpn_target_calculator=anchor_description.get_bbox_targets,
                anchor_description=anchor_description)

        if calc_bbox and (not self.training or calc_classes or calc_masks):
            roi_score, roi_bbox, roi_index = self.roi_selector(
                rpn_bbox, rpn_score, anchor_description)

            return_dict = dict(
                return_dict,
                roi_score=roi_score,
                roi_bbox=roi_bbox,
                roi_index=roi_index)
        elif self.training:
            roi_bbox = [self.default_roi_tensor for _ in range(batch_size)]

        if (calc_classes or calc_masks) and self.training:
            overlap_descriptions = self.overlap_calculator(roi_bbox, gt_bbox)
        else:
            overlap_descriptions = None

        if calc_classes and (self.training or calc_bbox):
            (
                mpn_class, mpn_class_selection, class_selector_description
            ) = self.class_network(
                class_feature_map, roi_bbox, overlap_descriptions)

            return_dict['mpn_class'] = mpn_class
            return_dict['mpn_class_coord_selection'] = mpn_class_selection
            return_dict['mpn_class_box_selection'] = class_selector_description

        if not self.training and calc_classes:
            if calc_bbox:
                assert class_selector_description is None
                (
                    return_dict['class'],
                    return_dict['class_propabilities'],
                    cat_class_indices
                ) = self.class_predictor(mpn_class, mpn_class_selection)

            if calc_gt_predictions and gt_bbox is not None:
                gt_bbox_class, gt_bbox_class_selection, _ = \
                    self.class_network(class_feature_map, gt_bbox, None)
                (
                    return_dict['gt_bbox_class'],
                    return_dict['gt_bbox_class_propabilities'],
                    gt_bbox_cat_class_indices
                ) = self.class_predictor(
                    gt_bbox_class, gt_bbox_class_selection)

        if calc_masks and (self.training or calc_bbox):
            (
                mpn_mask, mpn_mask_selection, mask_selector_description
            ) = self.mask_network(
                scene, mask_feature_map, unet_feature_maps, roi_bbox,
                overlap_descriptions)

            return_dict['mpn_mask'] = mpn_mask
            return_dict['mpn_mask_coord_selection'] = mpn_mask_selection
            return_dict['mpn_mask_box_selection'] = mask_selector_description

        if not self.training and calc_masks and calc_mask_predictions:
            if calc_bbox and calc_classes:
                assert mask_selector_description is None
                return_dict['mask'] = self.mask_predictor(
                    mpn_mask, mpn_mask_selection, cat_class_indices)

            if calc_gt_predictions and gt_bbox is not None and (
                    calc_classes or gt_label is not None):
                gt_bbox_mask, gt_bbox_mask_selection, _ = \
                    self.mask_network(
                        scene, mask_feature_map, unet_feature_maps,
                        gt_bbox, None)

                if calc_classes:
                    return_dict['gt_bbox_mask'] = self.mask_predictor(
                        gt_bbox_mask, gt_bbox_mask_selection,
                        gt_bbox_cat_class_indices)

                if gt_label is not None:
                    cat_gt_label = torch.cat(gt_label)
                    return_dict['gt_bbox_gt_label_mask'] = \
                        self.mask_predictor(
                            gt_bbox_mask, gt_bbox_mask_selection,
                            cat_gt_label)

        return return_dict


class DenseInputStage(nn.Module):
    def forward(self, scene):
        return torch.tensor(scene.shape[2:]), len(scene), scene


class SparseInputStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = CustomInputLayer(mode=4)

    def forward(self, data):
        scene = self.input_layer(*data[:4])
        spatial_size = data[2]
        batch_size = scene.batch_size()

        return spatial_size, batch_size, scene


class FeatureExtractor(nn.Module):
    def __init__(
            self, num_dims, sparse, input_channels, network_description,
            class_output_index=-1, class_output_upsampled=False,
            class_output_anchor=False, num_units=2, bottleneck_divisor=4, 
            stride=2, maxpool=False, num_dilations=2, use_residuals=True,
            main_path_relu=True, bottleneck_groups=1, drop_input_relu=False,
            relu_first=False, batchnorm=False,
            include_unet=False, unet_params=dict()):
        super().__init__()
        self.class_output_index = class_output_index
        self.class_output_upsampled = class_output_upsampled
        self.class_output_anchor = class_output_anchor
        self.num_dims = num_dims
        self.sparse = sparse
        self.input_channels = input_channels

        type_dict = dict(
            B=(
                get_units_level,
                dict(
                    num_units=num_units,
                    use_residuals=use_residuals,
                    bottleneck_divisor=bottleneck_divisor,
                    groups=bottleneck_groups,
                    stride=stride,
                    maxpool=maxpool,
                    make_dense=False,
                    drop_input_relu=drop_input_relu,
                    main_path_relu=main_path_relu,
                    relu_first=relu_first,
                    batchnorm=batchnorm)),
            D=(
                get_dilation_network,
                dict(
                    num_dilations=num_dilations,
                    make_dense=True,
                    kernel_size=3, dilation=1)),
            I=(
                get_identity,
                dict()),
            C=(
                get_sparse_to_dense,
                dict()),
        )

        # input stage
        self.input_stage = SparseInputStage() if sparse else DenseInputStage()

        # main network
        (
            self.sparse_indicator, stride_levels, self.channels_array,
            self.main_network
        ) = get_sequential_interims_layer(
            num_dims, sparse, input_channels, network_description,
            type_dict)

        self.cum_stride_levels = np.cumprod(stride_levels, axis=0)

        # anchor networks
        anchor_networks = [
            (index, layer_description.anchor_path)
            for index, layer_description
            in enumerate(network_description)
            if layer_description.anchor_path is not None]

        if len(anchor_networks):
            anchor_indices, anchor_network_descriptions = zip(*anchor_networks)
            self.anchor_indices = np.array(anchor_indices)

            anchor_input_sparse_indicator = self.sparse_indicator[
                self.anchor_indices]
            anchor_input_strides = self.cum_stride_levels[self.anchor_indices]
            anchor_input_channels = self.channels_array[self.anchor_indices]

            sparse_list, stride_list, channels_list, layer_list = zip(*(
                get_sequential_layer(
                    num_dims, anchor_sparse, anchor_channels,
                    anchor_network_description, type_dict)
                for (
                    anchor_sparse, anchor_channels, anchor_network_description)
                in zip(
                    anchor_input_sparse_indicator,
                    anchor_input_channels,
                    anchor_network_descriptions)
                ))
            self.anchor_sparse = np.array(sparse_list)
            self.anchor_strides = anchor_input_strides * stride_list
            self.anchor_channels = np.array(channels_list)
            self.anchor_networks = ZipApply(*layer_list)
        else:
            self.anchor_sparse = np.array([], dtype=bool)
            self.anchor_indices = np.array([], dtype=int)
            self.anchor_strides = self.cum_stride_levels[:0]
            self.anchor_channels = np.array([], dtype=int)
            self.anchor_networks = ZipApply()

        input_stride = np.full(num_dims, 1)
        main_stride_list = np.array([
            input_stride, *self.cum_stride_levels])
        main_sparse_list = np.array([sparse, *self.sparse_indicator])
        main_channels_list = np.array([input_channels, *self.channels_array])

        # mask selection
        correct_sparsity = main_sparse_list == self.sparse
        last_correct_sparsity = np.where(correct_sparsity)[0][-1]
        downsampling_indicator = (stride_levels != 1).any(-1)
        raw_skip_connection_indices, = np.where(
            downsampling_indicator[:last_correct_sparsity])
        self.skip_connection_strides = stride_levels[
            raw_skip_connection_indices]
        self.skip_connection_indices = np.array([
            *raw_skip_connection_indices, last_correct_sparsity])
        self.skip_connection_channels = main_channels_list[
            self.skip_connection_indices]

        if include_unet:
            unet_input_channels_list = self.skip_connection_channels[::-1]
            unet_input_stride_list = self.skip_connection_strides[::-1]
            (
                sparse_list, raw_unet_strides, self.unet_channels, self.unet
            ) = get_unet_upsampling_interims(
                num_dims, sparse, unet_input_channels_list,
                unet_input_stride_list, **unet_params)

            self.calc_unet = True

            
            self.unet_strides = (
                self.cum_stride_levels[-1] //
                np.cumprod(raw_unet_strides, axis=0))
        else:
            self.unet_strides, self.unet_channels, self.unet = None, None, None
            self.calc_unet = False

        if self.class_output_anchor:
            self.class_stride = self.anchor_strides[self.class_output_index]
            self.class_sparse = self.anchor_sparse[self.class_output_index]
            self.class_channels = self.anchor_channels[self.class_output_index]
        elif self.class_output_upsampled:
            assert include_unet
            self.class_stride = self.unet_strides[self.class_output_index]
            self.class_sparse = sparse_list[self.class_output_index]
            self.class_channels = self.unet_channels[self.class_output_index]
        else:
            self.class_stride = main_stride_list[self.class_output_index]
            self.class_sparse = main_sparse_list[self.class_output_index]
            self.class_channels = main_channels_list[self.class_output_index]

        self.mask_channels = [input_channels, *(
            main_channels_list[index]
            for index in self.skip_connection_indices)]

    def forward(self, scene):
        scene_size, batch_size, converted_scene = self.input_stage(scene)
        intermediate_outputs = self.main_network(converted_scene)
        anchor_inputs = [
            intermediate_outputs[index] for index in self.anchor_indices]
        anchor_outputs = self.anchor_networks(anchor_inputs)

        extended_intermediate_outputs = [
            converted_scene, *intermediate_outputs]

        mask_outputs = [
            extended_intermediate_outputs[index]
            for index in self.skip_connection_indices]

        if self.calc_unet:
            inverse_order = mask_outputs[::-1]
            unet_output = self.unet(
                inverse_order[0], inverse_order[1:])
        else:
            unet_output = None

        if self.class_output_anchor:
            class_output = anchor_outputs[
                self.class_output_index]
        elif not self.class_output_upsampled:
            class_output = extended_intermediate_outputs[
                self.class_output_index]
        else:
            class_output = unet_output[self.class_output_index]

        return (
            scene_size, batch_size, anchor_outputs, class_output, mask_outputs,
            unet_output)


class SegmentationNetwork(nn.Module):
    def __init__(
            self, num_dims, sparse, stride_list, channels_list, num_classes):
        super().__init__()

        channels = channels_list[-1]
        (
            self.sparse, stride, channels, self.channel_changer
        ) = get_channel_changer(
            num_dims, sparse, channels, num_classes)

        self.output_layer = (
            scn.OutputLayer(num_dims) if sparse else nn.Identity())

    def forward(self, x, scene):
        intermediate = x[-1]
        raw_output = self.channel_changer(intermediate)

        return self.output_layer(raw_output)


class ClassNetwork(nn.Module):
    def __init__(
            self, num_dims, sparse, input_channels, stride_prod,
            input_network_description, output_network_description,
            linear_channels=[256], num_classes=18, raw_scene=False,
            cut_shape=[4, 4, 4], pooling_function_or_none=torch.mean,
            relu_after_pooling=True, selection_tuple=None,
            *,
            positive_threshold, negative_threshold=0,):
        super().__init__()

        self.train_selector = TrainSelector(
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            random_selector=selection_tuple)

        type_dict = dict(
            D=(
                get_downsampler,
                dict()),
            S=(
                get_same_convolution,
                dict(kernel_size=3, dilation=1)),
            B=(
                get_units_level,
                dict(
                    use_residuals=True,
                    num_units=2,
                    bottleneck_divisor=4,
                    groups=1,
                    stride=1,
                    maxpool=False,
                    main_path_relu=False,
                    batchnorm=False)),
            I=(
                get_identity,
                dict()),
            M=(
                get_down_maxpooling,
                dict()),
            A=(
                get_down_avgpooling,
                dict()),
        )

        (
            sparse, stride, channels, self.input_conv_layer
        ) = get_sequential_layer_shape(
            num_dims, sparse, input_channels, input_network_description,
            type_dict, False)

        stride_prod = stride_prod * stride

        shape, self.roi_getter = get_roi_cut_or_align_layer(
            sparse, raw_scene=raw_scene, goal_shape=cut_shape,
            resize_boxes=stride_prod, clip_boxes=True)

        if not sparse and shape is None:
            print("!!!Warning: Slow single box computation not supported yet!")

        (
            sparse, stride, channels, self.output_conv_layer
        ) = get_sequential_layer_shape(
            num_dims, sparse, channels, output_network_description, type_dict,
            cut_shape is None)

        augmented_shape = None if shape is None else shape // stride

        channels, self.vectorice_layer = get_global_pool_or_reshape(
            sparse, num_dims, channels, augmented_shape,
            pooling_function=pooling_function_or_none)

        channel_list = [*linear_channels, num_classes]
        self.num_classes, self.linear_layer = get_linear_layer_network(
            channels, channel_list, start_relu=relu_after_pooling,
            end_relu=False)

        if not sparse and cut_shape is None:
            def empty_output(box_features):
                return box_features.new_zeros((0, num_classes))

            self.default_output_creator = empty_output

    def forward(self, feature_map, roi_bbox, overlap_descriptions):
        selected_bbox, selection_descriptions = (
            self.train_selector(overlap_descriptions)
            if self.training else
            (roi_bbox, None))

        if not len(selected_bbox) and self.default_output_creator is not None:
            return self.default_output_creator(feature_map)

        augmented_feature_map = self.input_conv_layer(feature_map)
        box_features, selection = self.roi_getter(
            augmented_feature_map, selected_bbox)
        convoluted_features = self.output_conv_layer(box_features)
        vectoriced_features = self.vectorice_layer(convoluted_features)
        class_scores = self.linear_layer(vectoriced_features)

        return class_scores, selection, selection_descriptions


class SparseMaskNetwork(nn.Module):
    class SparseFeaturemapSelectorBoth(nn.Module):
        def __init__(self, num_dims, spatial_size_extention):
            super().__init__()
            self.output_layer = scn.OutputLayer(num_dims)
            self.output_roi_cut = SparseRoiCut(
                RawToTensorFeatureExtractorCombiner())
            self.scene_roi_extra_cut = SparseRoiExtraCut(
                RawToFeaturesSceneFeatureExtractorCombiner())
            self.spatial_size_extention = spatial_size_extention

        def forward(self, raw_scene, feature_tensor, selected_bbox):
            coords, features, spatial_size, *other, batch_splits = raw_scene
            modified_spatial_size = spatial_size + self.spatial_size_extention
            output_features = self.output_layer(feature_tensor)
            combined_features = torch.cat((output_features, features), axis=-1)
            new_scene = (
                coords, combined_features, modified_spatial_size, *other,
                batch_splits)

            output_tensor, selection = self.output_roi_cut(
                new_scene, selected_bbox)
            skip_features = self.scene_roi_extra_cut(raw_scene, selection)

            return output_tensor, skip_features, selection
    class SparseFeaturemapSelector(nn.Module):
        def __init__(self, num_dims, spatial_size_extention):
            super().__init__()
            self.output_layer = scn.OutputLayer(num_dims)
            self.output_roi_cut = SparseRoiCut(
                RawToTensorFeatureExtractorCombiner())
            self.scene_roi_extra_cut = SparseRoiExtraCut(
                RawToFeaturesSceneFeatureExtractorCombiner())
            self.spatial_size_extention = spatial_size_extention

        def forward(self, raw_scene, feature_tensor, selected_bbox):
            coords, features, spatial_size, *other, batch_splits = raw_scene
            modified_spatial_size = spatial_size + self.spatial_size_extention
            output_features = self.output_layer(feature_tensor)
            new_scene = (
                coords, output_features, modified_spatial_size, *other,
                batch_splits)

            output_tensor, selection = self.output_roi_cut(
                new_scene, selected_bbox)
            skip_features = self.scene_roi_extra_cut(raw_scene, selection)

            return output_tensor, skip_features, selection

    class SparseFeaturemapSelectorRaw(nn.Module):
        def __init__(self, num_dims, spatial_size_extention):
            super().__init__()
            self.input_layer = CustomInputLayer(mode=4)
            self.roi_cut = SparseRoiCut(
                RawToRawFeatureExtractorCombiner())
            self.spatial_size_extention = spatial_size_extention

        def forward(self, raw_scene, feature_tensor, selected_bbox):
            selected_scene, selection = self.roi_cut(raw_scene, selected_bbox)
            coords, skip_features, spatial_size, batch_size = \
                selected_scene
            modified_spatial_size = spatial_size + self.spatial_size_extention
            output_tensor = self.input_layer(
                coords, skip_features, modified_spatial_size, batch_size)
            _, skip_features, *_ = selected_scene

            return output_tensor, skip_features, selection

    class SparseFeaturemapCombiner(nn.Module):
        def __init__(self, num_dims):
            super().__init__()
            self.output_layer = scn.OutputLayer(num_dims)

        def forward(self, feature_tensor, skip_features):
            forward_features = self.output_layer(feature_tensor)
            if len(forward_features):
                combined_features = torch.cat(
                    (forward_features, skip_features), axis=-1)
            else:
                combined_features = forward_features.new_zeros(
                    (0, forward_features.shape[1] + skip_features.shape[1]))
            return combined_features

    class SparseFeaturemapFirst(nn.Module):
        def __init__(self, num_dims):
            super().__init__()
            self.output_layer = scn.OutputLayer(num_dims)

        def forward(self, feature_tensor, skip_features):
            forward_features = self.output_layer(feature_tensor)
            return forward_features

    def __init__(
            self, num_dims, sparse, input_channels,
            skip_connection_channels, skip_connection_strides,
            unet_channels, unet_strides, *, use_unet_features,
            use_raw_features,
            use_skip_features, internal_unet=False, unet_params={},
            input_network_description, output_network_description,
            channel_list, selection_tuple=None, positive_threshold):
        super().__init__()

        type_dict = dict(
            S=(
                get_same_convolution,
                dict(
                    kernel_size=3,
                    dilation=1)),
            B=(
                get_units_level,
                dict(
                    use_residuals=True,
                    num_units=2,
                    bottleneck_divisor=4,
                    groups=1,
                    stride=1,
                    maxpool=False,
                    main_path_relu=False,
                    batchnorm=False)),
            I=(
                get_identity,
                dict())
        )

        self.train_selector = TrainSelector(
            positive_threshold=positive_threshold,
            negative_threshold=0,
            random_selector=selection_tuple)

        spatial_size_extention = torch.full((num_dims,), 32, dtype=torch.long)
        if use_unet_features:
            feature_channels = unet_channels[-1]
            (
                sparse, stride, channels, self.input_conv_layer
            ) = get_sequential_layer_shape(
                num_dims, sparse, feature_channels,
                input_network_description, type_dict, False)

            if use_raw_features:
                self.feature_map_selector = \
                    SparseMaskNetwork.SparseFeaturemapSelectorBoth(
                        num_dims, spatial_size_extention)
                channels += input_channels
            else:
                feature_channels = unet_channels[-1]
                self.feature_map_selector = \
                    SparseMaskNetwork.SparseFeaturemapSelector(
                        num_dims, spatial_size_extention)

            assert (stride == 1).all()
        else:
            assert use_raw_features
            self.input_conv_layer = None
            self.feature_map_selector = \
                SparseMaskNetwork.SparseFeaturemapSelectorRaw(
                    num_dims, spatial_size_extention)
            channels = input_channels


        assert sparse
        if internal_unet:
            sparse, stride, channels, self.output_conv_layer = get_unet(
                num_dims, sparse, channels, output_network_description,
                type_dict, False, **unet_params)
        else:

            (
                sparse, stride, channels, self.output_conv_layer
            ) = get_sequential_layer_shape(
                num_dims, sparse, channels, output_network_description,
                type_dict, False)

        assert (stride == 1).all()
        assert sparse

        if use_skip_features:
            self.output_feature_map_selector = \
                SparseMaskNetwork.SparseFeaturemapCombiner(num_dims)
            channels = channels + input_channels
        else:
            self.output_feature_map_selector = \
                SparseMaskNetwork.SparseFeaturemapFirst(num_dims)
            channels = channels
        self.classes, self.linear_layer = get_linear_layer_network(
            channels, channel_list, start_relu=False, end_relu=False)

    def forward(
            self, scene, mask_feature_map, unet_feature_maps, roi_bbox,
            overlap_descriptions):

        selected_bbox, selection_descriptions = (
            self.train_selector(overlap_descriptions)
            if self.training else
            (roi_bbox, None))
        if self.input_conv_layer is not None:
            feature_map = unet_feature_maps[-1]
            converted_input_features = self.input_conv_layer(feature_map)
        else:
            converted_input_features = None
        selected_input_features, selected_skip_features, selection = \
            self.feature_map_selector(
                scene, converted_input_features, selected_bbox)
        if len(selected_skip_features):
            output_feature_tensor = self.output_conv_layer(
                selected_input_features)
            output_features = self.output_feature_map_selector(
                output_feature_tensor, selected_skip_features)
            output = self.linear_layer(output_features)
        else:
            output = selected_skip_features.new_zeros((0, self.classes))
        return output, selection, selection_descriptions


class ClassPredictor(nn.Module):
    def forward(self, class_predictions, selection):
        _, bbox_sample_count, *_ = selection
        class_propabilities_raw = torch.softmax(class_predictions, 1)
        class_indices_raw = default_argmax(class_predictions, 1)
        class_propabilities = torch.split(
            class_propabilities_raw, bbox_sample_count)
        class_indices = torch.split(class_indices_raw, bbox_sample_count)
        return class_indices, class_propabilities, class_indices_raw


class DenseMaskPredictor(nn.Module):
    def __init__(self, num_valid=0):
        super().__init__()
        self.num_valid = num_valid

    def is_valid_class(self, single_class_to_select):
        is_valid = (0 <= single_class_to_select)
        if self.num_valid:
            is_valid &= (single_class_to_select < self.num_valid)
        return is_valid

    def create_mask(self, mpn_output, class_index, single_bbox, scene_size):
        if self.is_valid_class(class_index):
            detached_mask = mpn_output.detach()
            selected_logit_mask = detached_mask[class_index]
            selected_mask = torch.sigmoid(selected_logit_mask)
            return insert_mask(selected_mask, single_bbox, scene_size)
        else:
            return mpn_output.new_zeros(scene_size)

    def forward(self, mask_output, mask_selection, class_indices):
        bbox_tensor, bbox_sample_count, scene_size = mask_selection
        mask_tensor = torch.stack([
            self.create_mask(
                single_mask, single_class, single_bbox, scene_size)
            for single_mask, single_class, single_bbox
            in zip(mask_output, class_indices, bbox_tensor)])
        return torch.split(mask_tensor, bbox_sample_count)


class SparseMaskPredictor(nn.Module):
    def __init__(self, num_valid=0):
        super().__init__()
        self.num_valid = num_valid

    def is_valid_class(self, single_class_to_select):
        is_valid = (0 <= single_class_to_select)
        if self.num_valid:
            is_valid &= (single_class_to_select < self.num_valid)
        return is_valid

    def create_mask(self, mpn_output, mpn_class, single_bbox, scene_size):
        if self.is_valid_class(single_class_to_select):
            selected_logit_mask = mpn_output[mpn_class]
            selected_mask = torch.sigmoid(selected_logit_mask)
            return insert_mask(selected_mask, single_bbox, scene_size)
        else:
            return mpn_output.new_zeros(scene_size)

    def select_mask_and_sigmoid(self, splitted_masks, classes_to_select):
        return [
            torch.sigmoid(single_mpn_mask[single_class_to_select])
            if self.is_valid_class(single_class_to_select) else
            torch.zeros_like(single_mpn_mask[0])
            for single_mpn_mask, single_class_to_select
            in zip(splitted_masks, classes_to_select)]

    @staticmethod
    def get_full_coord_mask(selection, samplewise_mask):
        full_coord_masks = samplewise_mask.new_zeros(selection.shape)
        full_coord_masks[selection] = samplewise_mask
        return full_coord_masks

    def forward(self, mask_output, mask_selection, class_indices):
        is_inside, bbox_sample_count, batch_splits = mask_selection
        split_info = torch.tensor([bbox_sample_count, batch_splits])
        splitted_is_inside = split_select_nd(is_inside, split_info)

        new_coords_splits = is_inside.sum(1).tolist()
        sample_coord_splits = [
            sample_selection.sum().item()
            for sample_selection in splitted_is_inside]

        masks = mask_output.detach().T

        raw_boxwise_masks = torch.split(masks, new_coords_splits, dim=-1)
        selected_masks = torch.cat(
            self.select_mask_and_sigmoid(raw_boxwise_masks, class_indices))

        sample_wise_masks = torch.split(selected_masks, sample_coord_splits)

        mask_tensors = [
            self.get_full_coord_mask(sample_selection, sample_masks)
            for sample_selection, sample_masks
            in zip(splitted_is_inside, sample_wise_masks)]

        return mask_tensors


class SegmentationPredictor(nn.Module):
    def __init__(self, sparse):
        super().__init__()
        self.classes_dim = -1 if sparse else 0

    def forward(self, tensor):
        predicted_class = torch.argmax(tensor, self.classes_dim)
        predicted_probabilites = torch.softmax(tensor, self.classes_dim)
        return predicted_class, predicted_probabilites


class OverlapCalculator(nn.Module):
    @staticmethod
    def max_with_default(tensor):
        if tensor.numel():
            return tensor.max(1)
        else:
            return (
                tensor.new_zeros((tensor.shape[0],)),
                tensor.new_zeros((tensor.shape[0],), dtype=torch.long))

    @staticmethod
    def calc_overlap(pred_bbox, gt_bbox):
        box_overlap = bbox_overlap(
            pred_bbox.detach(), gt_bbox)
        return OverlapCalculator.max_with_default(box_overlap)

    def forward(self, pred_bbox_list, gt_bbox_list):
        return [
                (pred_bbox, gt_bbox, *self.calc_overlap(pred_bbox, gt_bbox))
                for pred_bbox, gt_bbox
                in zip(pred_bbox_list, gt_bbox_list)]


class TrainSelector(nn.Module):
    SelectionDescriptor = namedtuple(
        'SelectionDescriptor',
        ['forward_boxes', 'pred_selection', 'gt_selection', 'gt_association'])

    @staticmethod
    def all_selector(*args):
        return tuple(torch.arange(num) for num in args)

    def __init__(
            self,
            positive_threshold,
            negative_threshold=0,
            random_selector=None):
        super().__init__()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        if random_selector is None:
            self.random_selector = all_selector
        elif isinstance(random_selector, tuple):
            num_pos, num_neg, use_gt = random_selector
            if use_gt:
                def part_selector(pred, neg, gt):
                    pred_sel = torch.tensor(
                        np.random.choice(
                            pred, min(pred, num_pos), replace=False))
                    neg_sel = torch.tensor(
                        np.random.choice(
                            neg, min(neg, num_neg), replace=False))
                    gt_sel = torch.arange(gt)

                    return pred_sel, neg_sel, gt_sel
                self.random_selector = part_selector
            else:
                def part_selector(pred, neg, gt):
                    pred_sel = torch.tensor(
                        np.random.choice(
                            pred, min(pred, num_pos), replace=False))
                    neg_sel = torch.tensor(
                        np.random.choice(
                            neg, min(neg, num_neg), replace=False))
                    gt_sel = torch.arange(0)

                    return pred_sel, neg_sel, gt_sel
                self.random_selector = part_selector

    def select_single_sample(
            self, pred_bbox, gt_bbox, max_overlap, argmax_overlap):
        positive_unselected, = torch.where(
            max_overlap >= self.positive_threshold)
        if self.negative_threshold:
            negative_unselected, = torch.where(
                max_overlap < self.negative_threshold)
        else:
            negative_unselected = positive_unselected.new_zeros((0,))

        (
            selected_positive_index_indices,
            selected_negative_index_indices,
            selected_gt_indices_raw
        ) = self.random_selector(
            len(positive_unselected),
            len(negative_unselected),
            len(gt_bbox))

        selected_gt_indices = selected_gt_indices_raw.to(argmax_overlap.device)
        use_boxes_positive = positive_unselected[
            selected_positive_index_indices]
        use_boxes_negative = negative_unselected[
            selected_negative_index_indices]
        positive_gt_association = argmax_overlap[use_boxes_positive]
        negative_gt_association = argmax_overlap.new_full(
            use_boxes_negative.shape, -1)

        prediction_selection = torch.cat((
            use_boxes_positive, use_boxes_negative))
        gt_association = torch.cat((
            positive_gt_association,
            negative_gt_association,
            selected_gt_indices))

        forward_boxes = torch.cat((
            pred_bbox[prediction_selection],
            gt_bbox[selected_gt_indices]))

        return forward_boxes, self.SelectionDescriptor(
            forward_boxes, prediction_selection, selected_gt_indices,
            gt_association)

    def forward(self, max_argmax_pred_gt_tuple_list):
        forward_boxes_list, selection_descriptor_list = zip(*(
            self.select_single_sample(
                pred_bbox, gt_bbox, max_overlap, argmax_overlap)
            for pred_bbox, gt_bbox, max_overlap, argmax_overlap
            in max_argmax_pred_gt_tuple_list))
        return forward_boxes_list, selection_descriptor_list


class LossFilter(nn.Module):
    def __init__(self, positive_threshold, negative_threshold=0):
        super().__init__()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def forward(self, max_overlap, argmax_overlap):
        gt_association_raw = argmax_overlap
        keep = max_overlap >= self.positive_threshold
        if self.negative_threshold:
            negative = max_overlap < self.negative_threshold
            keep |= negative
            gt_association_raw = gt_association_raw.clone()
            gt_association_raw[negative] = -1
        gt_association = gt_association_raw[keep]
        return keep, gt_association


class ClassLossSelector(nn.Module):
    def __init__(
            self, positive_threshold, negative_threshold=0,
            negative_label=-100):
        super().__init__()
        self.loss_filter = LossFilter(positive_threshold, negative_threshold)
        self.negative_label = negative_label

    def single_sample_selection(self, gt_association, gt_labels):
        padded_labels = nn.functional.pad(
            gt_labels, (0, 1), value=self.negative_label)
        return padded_labels[gt_association]

    def forward(
            self, class_scores, selection, class_selector_description_list,
            pred_gt_max_argmax_tuple_list, gt_labels_list):
        _, box_sample_count, *_ = selection
        if class_selector_description_list is None:
            keep_list, gt_associations_list = zip(*(
                self.loss_filter(
                    max_overlap, argmax_overlap)
                for pred_bbox, gt_bbox, max_overlap, argmax_overlap
                in pred_gt_max_argmax_tuple_list))

            class_scores_samplewise = torch.split(
                class_scores, box_sample_count)

            selected_class_scores = [
                sample_class_score[keep]
                for sample_class_score, keep
                in zip(class_scores_samplewise, keep_list)]
        else:
            selected_class_scores = torch.split(class_scores, box_sample_count)
            gt_associations_list = (
                sample_class_selector_description.gt_association
                for sample_class_selector_description
                in class_selector_description_list)

        selected_labels = [
            self.single_sample_selection(gt_associations, gt_labels)
            for gt_associations, gt_labels
            in zip(gt_associations_list, gt_labels_list)]
        return selected_class_scores, selected_labels


class DenseMaskLossSelector(nn.Module):
    def __init__(self, positive_threshold):
        super().__init__()
        self.loss_filter = LossFilter(positive_threshold, 0)

    def forward(
            self, mask_scores, selection, class_selector_description_list,
            pred_gt_max_argmax_tuple_list, gt_labels_list, gt_masks_list):
        pred_mask_boxes_cat, box_sample_count, *_ = selection

        pred_mask_boxes_raw = torch.split(
            pred_mask_boxes_cat, box_sample_count)
        if class_selector_description_list is None:
            keep_list, gt_associations_list = zip(*(
                self.loss_filter(
                    max_overlap, argmax_overlap)
                for pred_bbox, gt_bbox, max_overlap, argmax_overlap
                in pred_gt_max_argmax_tuple_list))

            keep = torch.cat(keep_list)
            assert len(keep) == len(mask_scores)
            selected_mask_scores = [
                single_mask_score
                for single_mask_score, single_keep
                in zip(mask_scores, keep)
                if single_keep]

            pred_mask_boxes = [
                sample_mask_boxes[sample_keep]
                for sample_mask_boxes, sample_keep
                in zip(pred_mask_boxes_raw, keep_list)]
            splits = [len(sample_masks) for sample_masks in pred_mask_boxes]
        else:
            splits = box_sample_count
            selected_mask_scores = mask_scores
            pred_mask_boxes = pred_mask_boxes_raw
            gt_associations_list = [
                sample_class_selector_description.gt_association
                for sample_class_selector_description
                in class_selector_description_list]

        selected_labels = [
            gt_labels[gt_associations]
            for gt_associations, gt_labels
            in zip(gt_associations_list, gt_labels_list)]

        selected_labels_cat = torch.cat(selected_labels)

        selected_gt_masks = [
            gt_masks[gt_associations]
            for gt_associations, gt_masks
            in zip(gt_associations_list, gt_masks_list)]

        assert len(selected_labels_cat) == len(selected_mask_scores)

        selected_pred_masks = [
            single_mask[single_label]
            for single_mask, single_label
            in zip(selected_mask_scores, selected_labels_cat)]

        gt_masks = [
            [
                select_mask(single_gt_mask, single_bbox)
                for single_gt_mask, single_bbox
                in zip(sample_gt_mask, sample_bbox)]
            for sample_gt_mask, sample_bbox
            in zip(selected_gt_masks, pred_mask_boxes)]

        pred_masks = split_list(selected_pred_masks, splits)
        return pred_masks, gt_masks, selected_labels


class SparseMaskLossSelector(nn.Module):
    def __init__(self, positive_threshold):
        super().__init__()
        self.loss_filter = LossFilter(positive_threshold, 0)

    def forward(
            self, mask_scores, selection, class_selector_description_list,
            pred_gt_max_argmax_tuple_list, gt_labels_list, gt_masks_list):
        is_inside, box_sample_count, batch_splits = selection
        split_info = torch.tensor([box_sample_count, batch_splits])
        splitted_is_inside_raw = split_select_nd(is_inside, split_info)

        new_coords_splits = is_inside.sum(1).tolist()
        mask_scores_splitted = torch.split(
            mask_scores, new_coords_splits)

        if class_selector_description_list is None:
            keep_list, gt_associations_list = zip(*(
                self.loss_filter(
                    max_overlap, argmax_overlap)
                for pred_bbox, gt_bbox, max_overlap, argmax_overlap
                in pred_gt_max_argmax_tuple_list))

            keep = torch.cat(keep_list)

            assert len(keep) == len(mask_scores_splitted)
            selected_mask_scores = [
                single_mask_score
                for single_mask_score, single_keep
                in zip(mask_scores_splitted, keep)
                if single_keep]

            splitted_is_inside = [
                sample_is_inside[sample_keep]
                for sample_is_inside, sample_keep
                in zip(splitted_is_inside_raw, keep_list)]

            splits = [sample_keep.sum().item() for sample_keep in keep_list]
        else:
            splits = box_sample_count
            splitted_is_inside = splitted_is_inside_raw
            selected_mask_scores = mask_scores_splitted
            gt_associations_list = [
                sample_class_selector_description.gt_association
                for sample_class_selector_description
                in class_selector_description_list]

        selected_labels = [
            gt_labels[gt_associations]
            for gt_associations, gt_labels
            in zip(gt_associations_list, gt_labels_list)]

        selected_labels_cat = torch.cat(selected_labels)

        selected_gt_masks = [
            gt_masks[gt_associations]
            for gt_associations, gt_masks
            in zip(gt_associations_list, gt_masks_list)]

        gt_masks = [
            [
                single_mask[single_is_inside]
                for single_mask, single_is_inside
                in zip(sample_mask, sample_is_inside)]
            for sample_mask, sample_is_inside
            in zip(selected_gt_masks, splitted_is_inside)]

        assert len(selected_labels_cat) == len(selected_mask_scores)

        selected_pred_masks = [
            single_mask[:, single_label]
            for single_mask, single_label
            in zip(selected_mask_scores, selected_labels_cat)]

        pred_masks = split_list(selected_pred_masks, splits)
        return pred_masks, gt_masks, selected_labels
