import torch
import multiprocessing as mp
from torch.utils import data
from ndsis.data.sparse_augmentation import convert_sample

def load_sample(x):
    scene_id, scene_path = x
    return (scene_id,) + torch.load(scene_path)


def get_scene_ids(scene_file_list_filename):
    with open(scene_file_list_filename) as scene_file_list_file:
        return [line.rstrip('\n') for line in scene_file_list_file]


class AugmentedWorkingMemoryDataset(data.Dataset):
    def __init__(
            self, scene_id_list, raw_file_path, num_workers=0,
            **conversion_kwargs):
        scene_id_and_file_list = [
            (scene_id, raw_file_path.format(scene_id))
            for scene_id in scene_id_list]

        self.files = list(torch.utils.data.DataLoader(
            scene_id_and_file_list,
            collate_fn=lambda x: load_sample(x[0]),
            num_workers=num_workers))

        self.conversion_kwargs = conversion_kwargs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        return convert_sample(file, **self.conversion_kwargs)


class AugmentedStorageDataset(data.Dataset):
    def __init__(
            self, scene_id_list, raw_file_path, **conversion_kwargs):
        self.scene_id_and_file_list = [
            (scene_id, raw_file_path.format(scene_id))
            for scene_id in scene_id_list]
        self.conversion_kwargs = conversion_kwargs

    def __len__(self):
        return len(self.scene_id_and_file_list)

    def __getitem__(self, index):
        file = load_sample(self.scene_id_and_file_list[index])
        return convert_sample(file, **self.conversion_kwargs)


class WorkingMemoryDataset(data.Dataset):
    def __init__(
            self, scene_id_list, raw_file_path, num_workers=0):
        scene_id_and_file_list = [
            (scene_id, raw_file_path.format(scene_id))
            for scene_id in scene_id_list]

        self.files = list(torch.utils.data.DataLoader(
            scene_id_and_file_list,
            collate_fn=lambda x: load_sample(x[0]),
            num_workers=num_workers))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]


class StorageDataset(data.Dataset):
    def __init__(
            self, scene_id_list, raw_file_path):
        self.scene_id_and_file_list = [
            (scene_id, raw_file_path.format(scene_id))
            for scene_id in scene_id_list]

    def __len__(self):
        return len(self.scene_id_and_file_list)

    def __getitem__(self, index):
        return load_sample(self.scene_id_and_file_list[index])


def collate_fn(sample_list):
    (
        scene_id_list, coords_list, features_list, instance_bbox_tensor_list,
        shifted_instance_coords_list_list, semantic_instance_labels_list,
        semantic_segmentation_labels_list, augmentation_list, size_list
    ) = zip(*sample_list)

    coords_batch = torch.cat([
        torch.nn.functional.pad(coords, (0, 1), value=index)
        for index, coords in enumerate(coords_list)
    ]).cpu()

    batch_splits = [len(coords) for coords in coords_list]
    features_batch = torch.cat(features_list)
    semantic_labels_batch = torch.cat(semantic_segmentation_labels_list)
    spatial_size = torch.stack(size_list).max(0).values
    batch_size = len(scene_id_list)
    data = coords_batch, features_batch, spatial_size, batch_size, batch_splits

    return dict(
        id=scene_id_list,
        data=data,
        gt_bbox=instance_bbox_tensor_list,
        gt_label=semantic_instance_labels_list,
        gt_mask=shifted_instance_coords_list_list,
        gt_segmentation=semantic_labels_batch,
        batch_splits=batch_splits,
        augmentation=augmentation_list)


class AugmentedDataloader():
    def __init__(self, data_loader, **conversion_kwargs):
        self.data_loader = data_loader
        self.conversion_kwargs = conversion_kwargs
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            augmented_gen = [
                convert_sample(sample, **self.conversion_kwargs)
                for sample in batch]
            yield collate_fn(augmented_gen)


class PreaugmentDataset():
    def __init__(self, scene_list, raw_file_path, epochs):
        self.epochs = max(epochs, 1)
        self.epoch_index = 0
        self.len = len(scene_list)
        if epochs:
            self.data_set_list = [
                StorageDataset(scene_list, raw_file_path.format(epoch))
                for epoch in range(epochs)]
        else:
            self.data_set_list = [
                StorageDataset(scene_list, raw_file_path)]

    def __len__(self):
        return self.len

    def set_epoch(self, epoch):
        self.epoch_index = epoch % self.epochs

    def __getitem__(self, index):
        return self.data_set_list[self.epoch_index][index]


def get_data_loader(
        scene_list, raw_file_path, *, batch_size=1, num_workers=0,
        shuffle=False, preload=True, data_slice=slice(None),
        collate_augmentation=False, pre_augmented=False, epochs=None,
        **conversion_kwargs):

    scene_list = (
        scene_list
        if isinstance(scene_list, list) else
        get_scene_ids(scene_list))

    scene_list = scene_list[data_slice]

    if collate_augmentation:
        dataset = (
            WorkingMemoryDataset(
                scene_list, raw_file_path)
            if preload else
            StorageDataset(
                scene_list, raw_file_path))

        data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            shuffle=shuffle)

        return AugmentedDataloader(data_loader, **conversion_kwargs)            
    else:
        if pre_augmented:
            dataset = PreaugmentDataset(
                scene_list, raw_file_path, epochs)
        else:
            dataset = (
                AugmentedWorkingMemoryDataset(
                    scene_list, raw_file_path, **conversion_kwargs)
                if preload else
                AugmentedStorageDataset(
                    scene_list, raw_file_path, **conversion_kwargs))

        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle)


def batch_to_device(batch, device):
    batch = batch.copy()
    coords, colors, *remaining = batch['data']
    batch['data'] = coords, colors.to(device), *remaining
    batch['gt_bbox'] = [elem.to(device) for elem in batch['gt_bbox']]
    batch['gt_label'] = [elem.to(device) for elem in batch['gt_label']]
    batch['gt_mask'] = [elem.to(device) for elem in batch['gt_mask']]
    batch['gt_segmentation'] = batch['gt_segmentation'].to(device)
    return batch
