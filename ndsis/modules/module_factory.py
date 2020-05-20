from itertools import chain, repeat
import numpy as np
import torch
import torch.nn as nn
import sparseconvnet as scn
from ndsis.utils.basic_functions import product
from ndsis.modules.custom_operations import SparseGlobalPool
from ndsis.modules.roi_select_dense import RoiAlign, RoiCut as DenseRoiCut
from ndsis.modules.roi_select_sparse import (
    RawToTensorFeatureExtractorCombiner,
    TensorToTensorFeatureExtractorCombiner,
    SparseRoiCut)
from ndsis.modules.custom_container import (
    Elementwise, SequentialInterims, InverseSequentialInterims,
    ReuniteSequentialInterims, SkipConnectionReuniter, UnetContainer)


def get_dense_conv_class(num_dims):
    return [nn.Conv1d, nn.Conv2d, nn.Conv3d][num_dims - 1]


def get_dense_transposed_conv_class(num_dims):
    return [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][
        num_dims - 1]


def get_dense_maxpool_class(num_dims):
    return [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][num_dims - 1]


def get_dense_avgpool_class(num_dims):
    return [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][num_dims - 1]


def get_dense_batchnorm_class(num_dims):
    return [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][num_dims - 1]


class DenseResidual(nn.Module):
    def __init__(self, inner_block, residual_changer, activation=None):
        super().__init__()
        self.residual_changer = residual_changer
        self.inner_block = inner_block
        self.activation = nn.Identity() if activation is None else activation

    def forward(self, x):
        output = self.inner_block(x) + self.residual_changer(x)
        return self.activation(output)


def sparse_residual(inner_block, residual_changer, activation):
    layer = scn.Sequential(
        scn.ConcatTable(residual_changer, inner_block),
        scn.AddTable())
    if activation is not None:
        layer.append(activation)
    return layer


def get_residual(
        num_dims, sparse, input_channels, output_channels=None, *,
        inner_block, activation, residual_changer_bias=True):
    inner_sparse, stride, channels, inner_layer = inner_block
    assert inner_sparse == sparse
    if output_channels is not None:
        assert channels == output_channels

    if activation is None:
        activation_layer = None
    else:
        sparse, stride, channels, activation_layer = activation

    sparse, stride, channels, changer_layer = get_channel_changer_or_identity(
        num_dims, sparse, input_channels, channels)

    if sparse:
        residual_block = sparse_residual(
            inner_layer, changer_layer, activation_layer)
    else:
        residual_block = DenseResidual(
            inner_layer, changer_layer, activation_layer)

    return sparse, stride, channels, residual_block


def get_relu(num_dims, sparse, input_channels, inplace=True):
    stride = np.full(num_dims, 1)
    layer = (scn.ReLU() if sparse else nn.ReLU(inplace=inplace))
    return sparse, stride, input_channels, layer


def get_batchnorm_leaky_relu(
        num_dims, sparse, input_channels, *,
        eps=1e-4, momentum=0.9, leakiness=0):
    stride = np.full(num_dims, 1)
    if sparse:
        if leakiness:
            layer = scn.BatchNormLeakyReLU(
                input_channels, eps, momentum, leakiness)
        else:
            layer = scn.BatchNormReLU(
                input_channels, eps, momentum)
    else:
        batchnorm_class = get_dense_batchnorm_class(num_dims)
        batchnorm = batchnorm_class(input_channels, eps, momentum)

        if leakiness:
            relu = nn.LeakyReLU(leakiness, inplace=True)
        else:
            relu = nn.ReLU(inplace=True)

        layer = nn.Sequential(batchnorm, relu)
    return sparse, stride, input_channels, layer


def get_activation(
        num_dims, sparse, input_channels, batchnorm, inplace=True, *,
        eps=1e-4, momentum=0.9, leakiness=0):
    if batchnorm:
        return get_batchnorm_leaky_relu(
            num_dims, sparse, input_channels,
            eps=eps, momentum=momentum, leakiness=leakiness)
    else:
        return get_relu(num_dims, sparse, input_channels, inplace)


def get_residual_block(
        num_dims, sparse, input_channels, output_channels=None, *,
        groups=1, batchnorm=False, kernel_size=3, main_path_relu=False,
        relu_first=False, bottleneck_divisor=0, **kwargs):

    if output_channels is None:
        output_channels = input_channels

    if bottleneck_divisor:
        inner_channels = output_channels // bottleneck_divisor
        inner_block = (
            get_channel_changer(
                num_dims, sparse, input_channels, inner_channels,
                bias=not batchnorm),
            get_activation(
                num_dims, sparse, inner_channels, batchnorm, **kwargs),
            get_same_convolution(
                num_dims, sparse, inner_channels, inner_channels,
                kernel_size=kernel_size, groups=1, bias=not batchnorm),
            get_activation(
                num_dims, sparse, inner_channels, batchnorm, **kwargs),
            get_channel_changer(
                num_dims, sparse, inner_channels, output_channels,
                bias=not batchnorm))
    else:
        assert groups == 1
        inner_block = (
            get_same_convolution(
                num_dims, sparse, input_channels, output_channels,
                kernel_size=kernel_size, bias=not batchnorm),
            get_activation(
                num_dims, sparse, output_channels, batchnorm, **kwargs),
            get_same_convolution(
                num_dims, sparse, output_channels, output_channels,
                kernel_size=kernel_size, bias=not batchnorm))

    if main_path_relu or not relu_first:
        extra_activation = get_activation(
            num_dims, sparse, output_channels, batchnorm,
            inplace=True, **kwargs)
        if main_path_relu:
            output_activation = extra_activation
        else:
            inner_block = *inner_block, extra_activation
            output_activation = None
    else:
        extra_activation = get_activation(
            num_dims, sparse, input_channels, batchnorm,
            inplace=False, **kwargs)
        inner_block = extra_activation, *inner_block
        output_activation = None
    
    combined_inner_block = get_sequential(*inner_block)

    return get_residual(
        num_dims, sparse, input_channels,
        inner_block=combined_inner_block, activation=output_activation)


def get_convolution_blocks(
        num_dims, sparse, input_channels, output_channels, *, batchnorm=False,
        kernel_size=3, relu_first=False, num_units=1, **kwargs):

    input_channel_list_changers = chain(
        [input_channels], repeat(output_channels))

    if relu_first:
        input_channel_list_activations = chain(
            [input_channels], repeat(output_channels))
    else:
        input_channel_list_activations = repeat(output_channels)

    channel_changers = [
        get_same_convolution(
            num_dims, sparse, input_channels, input_channels,
            kernel_size=kernel_size, bias=(not batchnorm))
        for input_channels, _
        in zip(input_channel_list_changers, range(num_units))]

    activations = [
        get_activation(
            num_dims, sparse, input_channels, batchnorm,
            inplace=(index != 0 or not relu_first), **kwargs)
        for input_channels, index
        in zip(input_channel_list_activations, range(num_units))]

    if relu_first:
        layers = list(chain(*zip(activations, channel_changers)))
    else:
        layers = list(chain(*zip(channel_changers, activations)))

    return get_sequential(*layers)


def get_downsampler(
        num_dims, sparse, input_channels, output_channels,
        stride=2, bias=True):

    if isinstance(stride, int):
        stride = np.full(num_dims, stride)
    else:
        assert len(stride) == num_dims

    stride_tuple = tuple(stride)
    if sparse:
        layer = scn.Convolution(
            num_dims, input_channels, output_channels,
            filter_size=stride_tuple, filter_stride=stride_tuple, bias=bias)
    else:
        conv_class = get_dense_conv_class(num_dims)
        layer = conv_class(
            input_channels, output_channels, kernel_size=stride_tuple,
            stride=stride_tuple, padding=0, bias=bias)

    return sparse, stride, output_channels, layer


def get_upsampler(
        num_dims, sparse, input_channels, output_channels,
        stride=2, bias=True):

    if isinstance(stride, int):
        inverse_stride = np.full(num_dims, stride)
    else:
        inverse_stride = stride
        assert len(stride) == num_dims

    stride_tuple = tuple(inverse_stride)
    if sparse:
        layer = scn.Deconvolution(
            num_dims, input_channels, output_channels,
            filter_size=stride_tuple, filter_stride=stride_tuple, bias=bias)

    else:
        conv_class = get_dense_transposed_conv_class(num_dims)
        layer = conv_class(
            input_channels, output_channels, padding=0, bias=bias,
            kernel_size=stride_tuple, stride=stride_tuple)

    if (1 == inverse_stride).all():
        stride = inverse_stride
    else:
        stride = 1 / inverse_stride

    return sparse, stride, output_channels, layer


class DenseAddTable(nn.Module):
    def forward(self, x_list):
        output = torch.zeros_like(x[0])

        for x in x_list:
            output += x
        return output


class DenseJoinTable(nn.Module):
    def forward(self, x_list):
        output = torch.cat(x_list, axis=1)
        return output


def get_channels_combiner(
        num_dims, sparse, input_channels_list, concat=True,
        input_strideprod_list=None):

    if input_strideprod_list is not None:
        assert (
            input_strideprod_list[0] == np.array(input_strideprod_list)).all()

    stride = np.full(num_dims, 1)
    if concat:
        output_channels = sum(input_channels_list)
        if sparse:
            layer = scn.JoinTable()
        else:
            layer = DenseJoinTable()
    else:
        output_channels = input_channels_list[0]
        assert (np.array(input_channels_list) == output_channels).all()
        if sparse:
            layer = scn.AddTable()
        else:
            layer = DenseAddTable()

    return sparse, stride, output_channels, layer


def get_down_maxpooling(
        num_dims, sparse, input_channels, output_channels=None, *, stride=2):
    if isinstance(stride, int):
        stride = np.full(num_dims, stride)
    else:
        assert len(stride) == num_dims
    stride_tuple = tuple(stride)

    assert output_channels is None or output_channels == input_channels

    if sparse:
        layer = scn.MaxPooling(
            num_dims, pool_size=stride_tuple, pool_stride=stride_tuple)
    else:
        maxpool_class = get_dense_maxpool_class(num_dims)
        layer = maxpool_class(
            kernel_size=stride_tuple, stride=stride_tuple, padding=0)

    return sparse, stride, input_channels, layer


def get_down_avgpooling(
        num_dims, sparse, input_channels, output_channels=None, *, stride=2):
    if isinstance(stride, int):
        stride = np.full(num_dims, stride)
    else:
        assert len(stride) == num_dims
    stride_tuple = tuple(stride)

    assert output_channels is None or output_channels == input_channels

    if sparse:
        layer = scn.AveragePooling(
            num_dims, pool_size=stride_tuple, pool_stride=stride_tuple)
    else:
        maxpool_class = get_dense_avgpool_class(num_dims)
        layer = maxpool_class(
            kernel_size=stride_tuple, stride=stride_tuple, padding=0)

    return sparse, stride, input_channels, layer


def get_channel_changer_or_identity(
        num_dims, sparse, input_channels, output_channels=None,
        residual_changer_bias=True):
    stride = np.full(num_dims, 1)

    if output_channels is None or input_channels == output_channels:
        return get_identity(num_dims, sparse, input_channels)

    if sparse:
        layer = scn.NetworkInNetwork(
                    input_channels, output_channels, residual_changer_bias)
    else:
        conv_class = get_dense_conv_class(num_dims)
        layer = conv_class(
            input_channels, output_channels, kernel_size=1,
            bias=residual_changer_bias)

    return sparse, stride, output_channels, layer


def get_channel_changer(
        num_dims, sparse, input_channels, output_channels, kernel_size=1,
        bias=True):

    stride = np.full(num_dims, 1)
    if sparse:
        layer = scn.SubmanifoldConvolution(
            num_dims, input_channels, output_channels,
            filter_size=kernel_size, bias=bias)
    else:
        conv_class = get_dense_conv_class(num_dims)
        padding = np.array(kernel_size) // 2
        layer = conv_class(
            input_channels, output_channels,
            kernel_size=kernel_size, bias=bias, padding=padding)

    return sparse, stride, output_channels, layer


def get_same_convolution(
        num_dims, sparse, input_channels, output_channels=None, *,
        kernel_size, dilation=1, groups=1, bias=True):
    if output_channels is None:
        output_channels = input_channels
    stride = np.full(num_dims, 1)
    if sparse:
        assert dilation == 1
        layer = scn.SubmanifoldConvolution(
            num_dims, input_channels, output_channels,
            filter_size=kernel_size, bias=bias, groups=groups)
    else:
        conv_class = get_dense_conv_class(num_dims)
        layer = conv_class(
            input_channels, output_channels,
            kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias,
            dilation=dilation, groups=groups)

    return sparse, stride, output_channels, layer


def get_sequential(*stride_layer_list):
    sparse_list, strides, channels_list, layers = zip(*stride_layer_list)
    sparse = all(sparse_list)
    channels = channels_list[-1]
    sequential_class = scn.Sequential if sparse else nn.Sequential

    stride = np.prod(strides, axis=0)
    layer = sequential_class(*layers)

    return sparse, stride, channels, layer


def get_sparse_to_dense(
        num_dims, sparse, input_channels, output_channels=None):
    assert sparse
    assert input_channels == output_channels or output_channels is None
    stride = np.full(num_dims, 1)
    layer = scn.SparseToDense(num_dims, input_channels)
    return False, stride, input_channels, layer


def get_input_stage(
        num_dims, sparse, input_channels, output_channels=None, *,
        make_dense, stride, down=True, relu_first=True, drop_relu=False,
        maxpool=False, **kwargs):

    if output_channels is None:
        output_channels = input_channels

    input_stage = ()

    if relu_first and not drop_relu:
        input_stage += (get_activation(
            num_dims, sparse, input_channels, inplace=False,
            **kwargs),)

    if (np.array([stride]) == 1).all():
        input_stage += (
            get_channel_changer(
                    num_dims, sparse, input_channels, output_channels),)
    elif down:
        if maxpool:
            input_stage += (
                get_channel_changer(
                    num_dims, sparse, input_channels, output_channels),
                get_down_maxpooling(
                    num_dims, sparse, input_channels, stride=stride))
        else:
            input_stage += (
                get_downsampler(
                    num_dims, sparse, input_channels, output_channels,
                    stride),)
    else:
        assert not maxpool
        input_stage += (
            get_upsampler(
                num_dims, sparse, input_channels, output_channels,
                stride),)

    if not relu_first and not drop_relu:
        input_stage += (get_activation(
            num_dims, sparse, output_channels, inplace=True,
            **kwargs),)

    if sparse and make_dense:
        input_stage += (get_sparse_to_dense(
            num_dims, sparse, output_channels),)
        sparse = False
    
    return get_sequential(*input_stage)


def get_unit_stage(
        num_dims, sparse, input_channels, output_channels=None, *,
        num_units, use_residuals, **kwargs):

    if output_channels is None:
        output_channels = input_channels

    input_channel_list = chain([input_channels], repeat(output_channels))

    if use_residuals:
        units = (
            get_residual_block(
                num_dims, sparse, input_channels, output_channels, **kwargs)
            for input_channels, _ in zip(input_channel_list, range(num_units)))
        return get_sequential(*units)
    else:
        kwargs.pop('groups', None,)
        kwargs.pop('bottleneck_divisor', None)
        kwargs.pop('main_path_relu')
        return get_convolution_blocks(
            num_dims, sparse, input_channels, output_channels,
            num_units=num_units, **kwargs)


def get_units_level(
        num_dims, sparse, input_channels, output_channels=None, *,
        num_units, make_dense, stride, down=True, drop_input_relu=False,
        drop_input_stage=False,
        relu_first=True, maxpool=False, **kwargs):

    input_stage = get_input_stage(
        num_dims, sparse, input_channels, output_channels,
        make_dense=make_dense, stride=stride, down=down, relu_first=relu_first,
        maxpool=maxpool, **kwargs, drop_relu=drop_input_relu)

    sparse, _, channels, _ = input_stage

    unit_stage = get_unit_stage(
        num_dims, sparse, channels,
        relu_first=relu_first, num_units=num_units, **kwargs)

    return get_sequential(input_stage, unit_stage)


def get_skip_reunite_level(
        num_dims, sparse, input_channels, skip_channels, output_channels=None,
        *, stride, intermediate_channels=None,
        use_residuals, bottleneck_divisor, groups=1, main_path_relu,
        num_units, concat=True, relu_first=True, **kwargs):

    if output_channels is None:
        output_channels = skip_channels

    if concat:
        if intermediate_channels is None:
            intermediate_channels = output_channels
        concatenated_channels = skip_channels + intermediate_channels
    else:
        intermediate_channels = skip_channels
        concatenated_channels = skip_channels

    input_stage = get_input_stage(
        num_dims, sparse, input_channels, intermediate_channels,
        make_dense=False, stride=stride, down=False, relu_first=relu_first,
        **kwargs)

    sparse, stride, channels, input_layer = input_stage

    channels_combiner = get_channels_combiner(
        num_dims, sparse, [channels, skip_channels], concat)

    sparse, _, channels, combiner_layer = channels_combiner

    sparse,  _, channels, changer_layer = get_channel_changer_or_identity(
        num_dims, sparse, channels, output_channels)

    if num_units:
        sparse, _, channels, output_layer = get_unit_stage(
            num_dims, sparse, channels, use_residuals=use_residuals,
            bottleneck_divisor=bottleneck_divisor, groups=groups,
            main_path_relu=main_path_relu,
            relu_first=relu_first, num_units=num_units, **kwargs)
    else:
        sparse, _, channels, output_layer = get_identity(
            num_dims, sparse, channels)

    layer = SkipConnectionReuniter(
        input_layer, combiner_layer, changer_layer, output_layer)

    return sparse, stride, channels, layer


def get_dilation_network(
        num_dims, sparse, input_channels, output_channels=None, *,
        num_dilations, make_dense, depthwise_separable=False, **kwargs):

    layers = list()

    if sparse and make_dense:
        layers.append(get_sparse_to_dense(
            num_dims, sparse, input_channels, input_channels))
        sparse = False

    for _ in range(num_dilations):
        if depthwise_separable:
            layer = get_same_convolution(
                num_dims, sparse, input_channels, output_channels,
                kernel_size=1, dilation=1, groups=int(input_channels))
            layers.append(layer)
            sparse, _, input_channels, _ = layer
            layer = get_same_convolution(
                num_dims, sparse, input_channels, input_channels, **kwargs)
        else:
            layer = get_same_convolution(
                num_dims, sparse, input_channels, output_channels, **kwargs)

            layers.append(layer)

            sparse, _, input_channels, _ = layer

        layers.append(get_relu(num_dims, sparse, input_channels, inplace=True))

    return get_sequential(*layers)


def get_identity(num_dims, sparse, input_channels, output_channels=None):
    assert input_channels == output_channels or output_channels is None
    stride = np.full(num_dims, 1)
    identity = scn.Identity() if sparse else nn.Identity()
    return sparse, stride, input_channels, identity


class DenseGlobalPool(nn.Module):
    def __init__(self, num_dims, method):
        super().__init__()
        self.average_dims = tuple(range(-num_dims, 0))
        self.method = method

    def forward(self, x):
        return self.method(x, dim=self.average_dims)


class DenseListGlobalPool(nn.Module):
    def __init__(
            self, num_dims, method, input_channels,
            dtype=torch.get_default_dtype()):
        super().__init__()
        self.default_tensor = nn.Parameter(
            torch.tensor((0, input_channels), dtype=dtype),
            requires_grad=False)
        self.module = Elementwise(DenseGlobalPool(num_dims, method))

    def forward(self, x):
        samplewise_output = self.module(x)
        return (
            torch.stack(samplewise_output)
            if samplewise_output else
            self.default_tensor)


class Reshaper(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def get_global_pool(
        sparse, num_dims, input_channels, *, pooling_function, unequal_shape):
    if sparse:
        layer = SparseGlobalPool(pooling_function)
    elif unequal_shape:
        layer = DenseListGlobalPool(num_dims, pooling_function, input_channels)
    else:
        layer = DenseGlobalPool(num_dims, pooling_function)

    return input_channels, layer


def get_roi_cut_or_align_layer(
        sparse, *, raw_scene, goal_shape, resize_boxes, clip_boxes):
    assert not sparse or goal_shape is None
    if sparse:
        extractor_combiner = (
            RawToTensorFeatureExtractorCombiner()
            if raw_scene else
            TensorToTensorFeatureExtractorCombiner())
        layer = SparseRoiCut(
            extractor_combiner, clip_boxes=clip_boxes,
            resize_boxes=resize_boxes)
    elif goal_shape is None:
        layer = DenseRoiCut(clip_boxes=clip_boxes, resize_boxes=resize_boxes)
    else:
        layer = RoiAlign(
            goal_shape, clip_boxes=clip_boxes, resize_boxes=resize_boxes)

    return goal_shape, layer


def get_global_pool_or_reshape(
        sparse, num_dims, input_channels, input_shape, *, pooling_function):
    if pooling_function is None:
        assert not sparse
        layer = Reshaper()
        output_channels = input_channels * product(input_shape)
    else:
        output_channels, layer = get_global_pool(
            sparse, num_dims, input_channels, pooling_function=pooling_function,
            unequal_shape=(input_shape is None))

    return output_channels, layer


def get_linear_layer_network(
        input_channels, channel_list, *, start_relu, end_relu):
    layers = list()
    if start_relu:
        layers.append(nn.ReLU())
    first_output_channels = channel_list[0]
    layers.append(nn.Linear(input_channels, first_output_channels))
    input_channels = first_output_channels
    for channels in channel_list[1:]:
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(input_channels, channels))
        input_channels = channels

    if end_relu:
        layers.append(nn.ReLU(inplace=True))

    return input_channels, nn.Sequential(*layers)


def get_layers_inner(
        num_dims, sparse, input_channels, network_description, type_dict):
    layer_tuple_list = list()
    for layer_description in network_description:
        layer_class, default_kwargs = type_dict[layer_description.type]

        kwargs = {**default_kwargs, **layer_description.params}

        output_channels = layer_description.channels
        sparse, stride, input_channels, layers = layer_class(
            num_dims, sparse, input_channels, output_channels, **kwargs)

        layer_tuple_list.append((sparse, stride, input_channels, layers))

    return layer_tuple_list


def get_sequential_layer(
        num_dims, sparse, input_channels, network_description, type_dict):

    layer_tuple_list = get_layers_inner(
        num_dims, sparse, input_channels, network_description, type_dict)

    if len(layer_tuple_list):
        return get_sequential(*layer_tuple_list)
    else:
        return get_identity(num_dims, sparse, input_channels)


class UnequalShapeWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x_list):
        return [
            self.layer(x.unsqueeze(0)).squeeze(0)
            for x in x_list]


def get_sequential_layer_shape(
        num_dims, sparse, input_channels, network_description, type_dict,
        unequal_shape):
    sparse, stride, channels, layer = get_sequential_layer(
        num_dims, sparse, input_channels, network_description, type_dict)

    if unequal_shape and not sparse:
        layer = UnequalShapeWrapper(layer)

    return sparse, stride, channels, layer


def get_sequential_interims_layer(
        num_dims, sparse, input_channels, network_description, type_dict):

    layer_tuple_list = get_layers_inner(
        num_dims, sparse, input_channels, network_description, type_dict)

    sparse_list, stride_list, channels_list, layers_list = \
        zip(*layer_tuple_list)

    sparse_levels = np.array(sparse_list)
    stride_levels = np.array(stride_list)
    channels_levels = np.array(channels_list)
    layer = SequentialInterims(*layers_list)

    return sparse_levels, stride_levels, channels_levels, layer



def get_unet_upsampling(
        num_dims, sparse, input_channels_list, input_stride_list, *,
        min_channels=0, **kwargs):

    sequential_layers = list()
    channels = input_channels_list[0]
    skip_channels_list = input_channels_list[1:]
    output_channels_list = input_channels_list[1:]

    for skip_channels, output_channels, stride in zip(
            skip_channels_list, output_channels_list, input_stride_list):
        desired_channels = max(output_channels, min_channels)

        skip_layer = get_skip_reunite_level(
            num_dims, sparse, channels, skip_channels, desired_channels,
            stride=stride, **kwargs)

        sparse, _, channels, _ = skip_layer
        sequential_layers.append(skip_layer)

    sparse_list, strides, channels_list, layers = zip(*sequential_layers)

    sparse = all(sparse_list)
    channels = channels_list[-1]
    stride = np.prod(strides, axis=0)
    layer = InverseSequentialInterims(*layers)
    return sparse, stride, channels, layer


def get_unet_upsampling_interims(
        num_dims, sparse, input_channels_list, input_stride_list, *,
        min_channels=0, **kwargs):

    sequential_layers = list()
    channels = input_channels_list[0]
    skip_channels_list = input_channels_list[1:]
    output_channels_list = input_channels_list[1:]

    for skip_channels, output_channels, stride in zip(
            skip_channels_list, output_channels_list, input_stride_list):
        desired_channels = max(output_channels, min_channels)

        skip_layer = get_skip_reunite_level(
            num_dims, sparse, channels, skip_channels, desired_channels,
            stride=stride, **kwargs)

        sparse, _, channels, _ = skip_layer
        sequential_layers.append(skip_layer)

    sparse_list, strides, channels_list, layers = zip(*sequential_layers)

    sparse = np.array(sparse_list)
    channels = np.array(channels_list)
    stride = input_stride_list[::-1]
    layer = ReuniteSequentialInterims(*layers)
    return sparse, stride, channels, layer


def get_unet(
        num_dims, sparse, input_channels, network_description, type_dict,
        min_upsampling_channels, **upsampling_params):
    sparse_levels, stride_levels, channels_levels, downsampling_layer = \
        get_sequential_interims_layer(
            num_dims, sparse, input_channels, network_description, type_dict)

    sparse, stride, channels, upsampling_layer = get_unet_upsampling(
        num_dims, sparse, channels_levels[::-1], stride_levels[::-1],
        **upsampling_params)

    layer = UnetContainer(downsampling_layer, upsampling_layer)
    stride = np.full(num_dims, 1)
    return sparse, stride, channels, layer
