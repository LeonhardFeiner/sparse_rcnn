
import torch
import torch.nn as nn
import sparseconvnet as scn


def get_original_features(feature_map):
    return scn.ioLayers.OutputLayerFunction.apply(
      len(feature_map.spatial_size),
      feature_map.metadata, feature_map.features)


def create_sparse_tensor(spatial_size, coords, features):
    dimension = spatial_size.ndim
    metadata = scn.Metadata(dimension)

    sparse_features = scn.ioLayers.InputLayerFunction.apply(
        dimension, metadata, spatial_size, coords, features, 0, 4)

    return scn.SparseConvNetTensor(
        features=sparse_features, metadata=metadata, spatial_size=spatial_size)


def split_batch(sparse_tensor):
    batch_size = sparse_tensor.batch_size()
    batch_associations = sparse_tensor.get_spatial_locations()[:, -1]

    sparse_features = sparse_tensor.features

    batch_associations = sparse_tensor.get_spatial_locations()[:, -1]
    batch_range = torch.arange(batch_size).unsqueeze(1)
    sample_associations = batch_range == batch_associations

    associated_features = [
        sparse_features[is_associated]
        for is_associated
        in sample_associations]

    return associated_features


class SparseGlobalPool(nn.Module):
    def __init__(self, pooling_function=torch.mean):
        super().__init__()
        self.pooling_function = pooling_function

    def forward(self, sparse_tensor):
        associated_features = split_batch(sparse_tensor)
        if(associated_features):

            return torch.stack([
                self.pooling_function(sample_features, dim=0)
                if len(sample_features) else
                sample_features.new_zeros((sample_features.shape[1]))
                for sample_features
                in associated_features])

        else:
            return sparse_tensor.features[:0]


class CustomInputLayer(nn.Module):
    def __init__(self, mode=3):
        super().__init__()
        self.mode = mode

    def forward(self, coords, features, spatial_size, batch_size=0):
        spatial_size = torch.as_tensor(spatial_size, dtype=torch.long)
        dimension = len(spatial_size)
        metadata = scn.Metadata(dimension)
        if len(coords):
            sparse_features = scn.ioLayers.InputLayerFunction.apply(
                dimension,
                metadata,
                spatial_size,
                coords.long(),
                features,
                batch_size,
                self.mode
            )
            return scn.SparseConvNetTensor(
                metadata=metadata, spatial_size=spatial_size,
                features=sparse_features)

        else:
            return None
