import torch
import torch.nn as nn
import sparseconvnet as scn
from ndsis.utils.basic_functions import product, split_select_nd, is_sorted
from ndsis.modules.roi_select_bbox_transform import BBoxTransformerSlice


class SparseRoiExtraCut(nn.Module):
    def __init__(self, feature_extractor_combiner):
        super().__init__()
        self.feature_extractor_combiner = feature_extractor_combiner

    def forward(self, feature_map, selection, new_coords=None):
        is_inside, bbox_sample_count, batch_splits = selection
        old_coords, old_features, spatial_size, batch_splits = (
            self.feature_extractor_combiner.extract(feature_map))

        new_features = select_features(old_features, is_inside)

        if self.feature_extractor_combiner.NEED_COORDS and new_coords is None:
            new_coords = select_coords(old_coords, is_inside)

        box_features = self.feature_extractor_combiner.combine(
            new_coords, new_features, spatial_size, len(is_inside))

        return box_features


class SparseRoiCut(nn.Module):
    def __init__(
            self, feature_extractor_combiner,
            clip_boxes=False, resize_boxes=None):
        super().__init__()
        self.bbox_transformer = BBoxTransformerSlice(
            clip=clip_boxes, resize=resize_boxes)
        self.feature_extractor_combiner = feature_extractor_combiner

    def forward(self, feature_map, bbox_batch):
        old_coords, old_features, spatial_size, batch_splits = (
            self.feature_extractor_combiner.extract(feature_map))

        bbox_tensor, bbox_sample_count, bbox_sample_association = (
            self.bbox_transformer(bbox_batch, spatial_size))

        new_coords, new_features, is_inside = roi_cut(
            old_coords, old_features, bbox_tensor,
            bbox_sample_association.to(bbox_tensor.device))

        box_features = self.feature_extractor_combiner.combine(
            new_coords, new_features, spatial_size, len(is_inside))

        return box_features, (is_inside, bbox_sample_count, batch_splits)


class RawToFeaturesSceneFeatureExtractorCombiner():
    NEED_COORDS = False
    @staticmethod
    def extract(feature_map):
        new_coords, new_features, spatial_size, *_, batch_splits = feature_map
        return new_coords, new_features, spatial_size, batch_splits

    @staticmethod
    def combine(new_coords, new_features, spatial_size, batch_size=0):
        return new_features


class RawToTensorFeatureExtractorCombiner():
    NEED_COORDS = True
    @staticmethod
    def extract(feature_map):
        new_coords, new_features, spatial_size, *_, batch_splits = feature_map
        return new_coords, new_features, spatial_size, batch_splits

    @staticmethod
    def combine(new_coords, new_features, spatial_size, batch_size=0):
        ndim = len(spatial_size)
        metadata = scn.Metadata(ndim)

        features = scn.ioLayers.InputLayerFunction.apply(
            ndim, metadata, spatial_size, new_coords, new_features, batch_size,
            4)

        return scn.SparseConvNetTensor(
            features=features, metadata=metadata, spatial_size=spatial_size)


class RawToRawFeatureExtractorCombiner():
    NEED_COORDS = True
    @staticmethod
    def extract(feature_map):
        new_coords, new_features, spatial_size, *_, batch_splits = feature_map
        return new_coords, new_features, spatial_size, batch_splits

    @staticmethod
    def combine(new_coords, new_features, spatial_size, batch_size=0):
        return new_coords, new_features, spatial_size, batch_size


class TensorToTensorFeatureExtractorCombiner():
    NEED_COORDS = True
    @staticmethod
    def extract(feature_map):
        old_coords = feature_map.get_spatial_locations()
        old_features = feature_map.features
        spatial_size = feature_map.spatial_size
        batch_association = old_coords[:, -1]
        assert is_sorted(batch_association)
        batch_splits = torch.bincount(
            batch_association, minlength=feature_map.batch_size())
        return old_coords, old_features, spatial_size, batch_splits

    @staticmethod
    def combine(new_coords, new_features, spatial_size, batch_size=0):
        ndim = len(spatial_size)
        metadata = scn.Metadata(ndim)

        features = scn.ioLayers.InputLayerFunction.apply(
            ndim, metadata, spatial_size, new_coords, new_features, batch_size,
            0)

        return scn.SparseConvNetTensor(
            features=features, metadata=metadata, spatial_size=spatial_size)


def select_features(features, is_inside):
    num_boxes, num_coords = is_inside.shape
    if num_boxes:
        features_expanded = (
            features[None, :, :].expand(num_boxes, -1, -1))  # BBxNxC
        selected_features = features_expanded[is_inside]     # (BBxn)xC
    else:
        selected_features = torch.zeros_like(features[:0])
    return selected_features


def select_coords(coords, is_inside):
    num_boxes, num_coords = is_inside.shape
    coords_expanded = (
        coords[None, :, :-1].expand(num_boxes, -1, -1))  # BBxNxD
    box_index_expanded = (
        torch.arange(num_boxes, device=coords.device)
        [:, None, None].expand(-1, num_coords, 1))       # BBxNx1

    selected_coords = coords_expanded[is_inside]         # (BBxn)xD
    selected_box_index = box_index_expanded[is_inside]   # (BBxn)x1
    extended_coordinates = torch.cat(
        (selected_coords, selected_box_index), -1)       # (BBxn)x(D+1)

    return extended_coordinates


def inside_converter(is_inside, bbox_sample_count, batch_splits):
    split_info = torch.tensor([bbox_sample_count, batch_splits])
    return split_select_nd(is_inside, split_info)


def get_inside_indicator(coords, bbox_tensor, bbox_sample_association):
    expanded_bbox_sample_association = torch.stack(
        (bbox_sample_association, bbox_sample_association + 1), -1
        ).unsqueeze(-1)

    extended_boxes = torch.cat(
        (bbox_tensor, expanded_bbox_sample_association), dim=-1)
    start_dims, stop_dims = extended_boxes.unsqueeze(1).unbind(2)  # BBx1x(D+1)
    is_inside = (((start_dims <= coords) & (coords < stop_dims)).all(-1))

    return is_inside


def roi_cut(
        coords, features, bbox_tensor, bbox_sample_association):
    coords = coords.to(bbox_tensor.device)

    is_inside = get_inside_indicator(
        coords, bbox_tensor, bbox_sample_association)

    selected_features = select_features(features, is_inside)
    extended_coordinates = select_coords(coords, is_inside)

    return extended_coordinates.cpu(), selected_features, is_inside.cpu()
