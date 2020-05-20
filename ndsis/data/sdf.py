# %%
import torch
from torch.utils import data

from ndsis.utils.mask import insert_mask_list


def get_scene_ids(scene_file_list_filename):
    with open(scene_file_list_filename) as scene_file_list_file:
        return [line.rstrip('\n') for line in scene_file_list_file]


def load_sample(x):
    scene_id, scene_path = x
    return (scene_id,) + torch.load(scene_path)


def convert_sample_dense(x, truncate=3, **conversion_args):
    scene_id, data_raw, gt_label, gt_bbox, gt_mask = x

    trunc_data = torch.clamp(data_raw, -truncate, truncate)
    trunc_abs_data = torch.abs(trunc_data)
    greater_data = (data_raw > -1).type(data_raw.dtype)
    data = torch.stack([trunc_abs_data, greater_data])

    gt_mask_expanded = insert_mask_list(
        gt_mask, gt_bbox.long(), data_raw.shape)  # remove .long

    return scene_id, data, gt_label, gt_bbox.float(), gt_mask, gt_mask_expanded


def convert_sample_sparse(x):
    scene_id, data_raw, gt_label, gt_bbox, gt_mask = x

    occupied = torch.abs(data_raw) <= 1
    torch.where(occupied)

    gt_mask_expanded = insert_mask_list(
        gt_mask, gt_bbox.long(), data_raw.shape)  # remove .long

    gt_mask_occupied = gt_mask_expanded[:, occupied]
    if len(gt_mask_occupied):
        values, gt_indices = gt_mask_occupied.max(0)
        gt_indices[~values] = -1
    else:
        gt_indices = torch.full(
            gt_mask_occupied.shape[1:], -1, dtype=torch.long)

    coords = torch.stack(torch.where(occupied), axis=1)
    features = torch.ones((len(coords), 1))

    spatial_size = torch.tensor(data_raw.shape, dtype=torch.long)
    return (
        scene_id, coords, features, gt_bbox.float(), gt_indices, gt_label,
        spatial_size)


class StorageDataset(data.Dataset):
    def __init__(
            self, scene_id_list, raw_file_path, *, dense, **conversion_kwargs):
        self.scene_id_and_file_list = [
            (scene_id, raw_file_path.format(scene_id))
            for scene_id in scene_id_list]
        self.conversion_kwargs = conversion_kwargs

        self.convert_sample = (
            convert_sample_dense if dense else convert_sample_sparse)

    def __len__(self):
        return len(self.scene_id_and_file_list)

    def __getitem__(self, index):
        file = load_sample(self.scene_id_and_file_list[index])
        return self.convert_sample(file, **self.conversion_kwargs)


def collate_fn_dense(sample_list):
    (
        scene_id_list, data_list, gt_label_list, gt_bbox_list,
        gt_mask_cutout_list, gt_mask_expanded_list
    ) = zip(*sample_list)

    data = torch.stack(data_list)

    return dict(
        id=scene_id_list,
        data=data,
        gt_bbox=gt_bbox_list,
        gt_label=gt_label_list,
        gt_mask=gt_mask_expanded_list)

def collate_fn_sparse(sample_list):
    (
        scene_id_list, coords_list, features_list, instance_bbox_tensor_list,
        shifted_instance_coords_list_list, semantic_instance_labels_list,
        size_list
    ) = zip(*sample_list)

    coords_batch = torch.cat([
        torch.nn.functional.pad(coords, (0, 1), value=index)
        for index, coords in enumerate(coords_list)
    ]).cpu()

    batch_splits = [len(coords) for coords in coords_list]
    features_batch = torch.cat(features_list)
    spatial_size = torch.stack(size_list).max(0).values
    batch_size = len(scene_id_list)
    data = coords_batch, features_batch, spatial_size, batch_size, batch_splits

    return dict(
        id=scene_id_list,
        data=data,
        gt_bbox=instance_bbox_tensor_list,
        gt_label=semantic_instance_labels_list,
        gt_mask=shifted_instance_coords_list_list,
        batch_splits=batch_splits)


def batch_to_device(batch, device):
    batch = batch.copy()
    batch['data'] = batch['data'].to(device)
    batch['gt_bbox'] = [sample.to(device) for sample in batch['gt_bbox']]
    batch['gt_label'] = [sample.to(device) for sample in batch['gt_label']]
    batch['gt_mask'] = [
        sample.to(device) for sample in batch['gt_mask']]
    return batch


def batch_to_device_sparse(batch, device):
    batch = batch.copy()
    coords, colors, *remaining = batch['data']
    batch['data'] = coords, colors.to(device), *remaining
    batch['gt_bbox'] = [elem.to(device) for elem in batch['gt_bbox']]
    batch['gt_label'] = [elem.to(device) for elem in batch['gt_label']]
    batch['gt_mask'] = [elem.to(device) for elem in batch['gt_mask']]
    return batch


def get_data_loader(
        scene_list, raw_file_path, *, batch_size, num_workers,
        shuffle, data_slice=slice(None), dense, **conversion_kwargs):

    scene_list = (
        scene_list
        if isinstance(scene_list, list) else
        get_scene_ids(scene_list))

    scene_list = scene_list[data_slice]

    dataset = StorageDataset(
        scene_list, raw_file_path, dense=dense, **conversion_kwargs)

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_dense if dense else collate_fn_sparse,
        shuffle=shuffle)
