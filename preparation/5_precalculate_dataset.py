# %%
import context
import torch
import os
import multiprocessing as mp
from itertools import product
from scannet_config.run import (
    input_scale_neg_exponent,
    train_loader_params, val_loader_params,
    selected_train_path, selected_val_path,
    selected_train_scene_ids, selected_val_scene_ids)
from ndsis.data.data import AugmentedStorageDataset
from scannet_config.pathes import ( 
    preaugmented_path
#    scene_ids_train, scene_ids_val,
#    low_res_preprocessed_trainval, high_res_preprocesseds_trainval
)
# %%

def reduce_dict(
        *, batch_size, num_workers, shuffle, preload, pre_augmented,
        data_slice=None, collate_augmentation, device, **conversion_kwargs):
    return conversion_kwargs


if False:
    params = reduce_dict(**train_loader_params)
    path = selected_train_path
    scene_ids = selected_train_scene_ids
    name = 'train'
    epochs = 256
elif True:
    params = reduce_dict(**val_loader_params)
    path = selected_train_path
    scene_ids = selected_train_scene_ids[:len(selected_val_scene_ids)]
    name = 'trainval'
    epochs = None
else:
    params = reduce_dict(**val_loader_params)
    path = selected_val_path
    scene_ids = selected_val_scene_ids
    name = 'val'
    epochs = None

exist_ok = True
data_set = AugmentedStorageDataset(scene_ids, path, **params)
augmention_id = f'voxelsize0375_NYUi18s20'

# %%

output_path = preaugmented_path / augmention_id / name
os.makedirs(output_path, exist_ok=exist_ok)
if epochs is None:
    epoch_index_combination = range(len(data_set))
else:
    epoch_index_combination = list(
        product(range(epochs), range(len(data_set))))
    for epoch in range(epochs):
        (output_path / f'epoch{epoch:0>3}').mkdir(exist_ok=exist_ok)

torch.save(params, output_path / 'params.pth')

# %%

def convert_scene(epoch_index):
    if isinstance(epoch_index, tuple):
        epoch, index = epoch_index
        raw_path = output_path / f'epoch{epoch:0>3}'
    else:
        index = epoch_index
        raw_path = output_path

    scene_id, *scene = data_set[index]
    path = raw_path / f'{scene_id}.pth'

    torch.save(tuple(scene), path)
    print(scene_id, epoch_index)

# %%
if __name__ == '__main__':
    print('start_augmentation')
    list(torch.utils.data.DataLoader(
        epoch_index_combination,
        collate_fn=lambda x: convert_scene(x[0]),
        num_workers=4))#mp.cpu_count()))

# %%