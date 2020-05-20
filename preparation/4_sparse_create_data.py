# %%
import plyfile
import numpy as np
import torch
from multiprocessing import Pool
from functools import partial
import os

#import context
from ndsis.utils.mesh import get_vertex_parts

# %%


def get_colors_coords_normals(plyfile):
    coords = get_vertex_parts(plyfile, ['x', 'y', 'z'])
    raw_colors = get_vertex_parts(plyfile, ['red', 'green', 'blue'])
    raw_normals = get_vertex_parts(plyfile, ['nx', 'ny', 'nz'])

    colors = raw_colors / 127.5 - 1
    normal_length = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    normal_length[normal_length == 0] = 1
    normals = raw_normals / normal_length

    return coords, colors, normals


def save_torch_tuple(
        result_path, coords, colors, normals, instance_ids,
        semantic_instance_labels):
    result_tuple = (
        torch.from_numpy(coords).float(),
        torch.from_numpy(colors).float(),
        torch.from_numpy(normals).float(),
        torch.from_numpy(instance_ids),
        torch.from_numpy(semantic_instance_labels))

    torch.save(result_tuple, result_path)

def convert_scene_raw_augmented_alpha(
        raw_scene_path, raw_instance_labels_path, raw_result_path, scene_id):
    print(scene_id)
    scene_path = raw_scene_path.format(scene_id)
    instance_labels_path = raw_instance_labels_path.format(scene_id)
    result_path = raw_result_path.format(scene_id)

    semantic_instance_labels = np.load(instance_labels_path)
    orig_plyfile = plyfile.PlyData.read(scene_path)
    raw_instance_ids = get_vertex_parts(orig_plyfile,  ['alpha'])
    instance_ids = raw_instance_ids.astype(np.int8).astype(np.int).squeeze(-1)
    coords, colors, normals = get_colors_coords_normals(orig_plyfile)

    save_torch_tuple(
        result_path, coords, colors, normals, instance_ids,
        semantic_instance_labels)


def convert_scene_raw_id_file(
        raw_scene_path, raw_instance_ids_path, raw_instance_labels_path,
        raw_result_path, scene_id):
    print(scene_id)
    scene_path = raw_scene_path.format(scene_id)
    instance_ids_path = raw_instance_ids_path.format(scene_id)
    instance_labels_path = raw_instance_labels_path.format(scene_id)
    result_path = raw_result_path.format(scene_id)

    semantic_instance_labels = np.load(instance_labels_path)
    instance_ids = np.load(instance_ids_path)

    orig_plyfile = plyfile.PlyData.read(scene_path)
    coords, colors, normals = get_colors_coords_normals(orig_plyfile)

    save_torch_tuple(
        result_path, coords, colors, normals, instance_ids,
        semantic_instance_labels)


def convert_test_scene_raw_file(raw_scene_path, raw_result_path, scene_id):
    print(scene_id)
    scene_path = raw_scene_path.format(scene_id)
    result_path = raw_result_path.format(scene_id)
    orig_plyfile = plyfile.PlyData.read(scene_path)
    coords, colors, normals = get_colors_coords_normals(orig_plyfile)

    semantic_instance_labels = np.zeros(0, dtype=int)
    instance_ids = np.full(len(coords), -1)

    save_torch_tuple(
        result_path, coords, colors, normals, instance_ids,
        semantic_instance_labels)


# %%

def convert_scene_list_augmented_alpha(
        raw_scene_path, raw_instance_labels_path,
        raw_result_path, scene_ids, processes=None):

    if processes in (0, False):
        for scene_id in scene_ids:
            convert_scene_raw_augmented_alpha(
                raw_scene_path, raw_instance_labels_path,
                raw_result_path, scene_id)
    else:
        convert_scene = partial(
            convert_scene_raw_augmented_alpha, raw_scene_path,
            raw_instance_labels_path, raw_result_path)
        p = Pool(processes=processes)
        p.map(convert_scene, scene_ids)
        p.close()
        p.join()


def convert_scene_list_id_file(
        raw_scene_path, raw_instance_ids_path, raw_instance_labels_path,
        raw_result_path, scene_ids, processes=None):

    if processes in (0, False):
        for scene_id in scene_ids:
            convert_scene_raw_id_file(
                raw_scene_path, raw_instance_ids_path,
                raw_instance_labels_path, raw_result_path, scene_id)
    else:
        convert_scene = partial(
            convert_scene_raw_id_file, raw_scene_path, raw_instance_ids_path,
            raw_instance_labels_path, raw_result_path)
        p = Pool(processes=processes)
        p.map(convert_scene, scene_ids)
        p.close()
        p.join()

def convert_test_scene_list_file(
        raw_scene_path, raw_result_path, scene_ids, processes=None):

    if processes in (0, False):
        for scene_id in scene_ids:
            convert_test_scene_raw_file(
                raw_scene_path, raw_result_path, scene_id)
    else:
        convert_scene = partial(
            convert_test_scene_raw_file, raw_scene_path, raw_result_path)
        p = Pool(processes=processes)
        p.map(convert_scene, scene_ids)
        p.close()
        p.join()

# %%


def preprocess_id_file_pathes(high_res=False, test=False, processes=None):
    from scannet_config.pathes import (
        scene_ids_train, scene_ids_val, scene_ids_test,
        high_res_instance_ids, low_res_instance_ids,
        high_res_instance_labels, low_res_instance_labels,
        low_res_extended_ply_files_trainval, low_res_extended_ply_files_test,
        high_res_extended_ply_files_trainval, high_res_extended_ply_files_test,
        low_res_preprocessed_trainval, low_res_preprocessed_test,
        high_res_preprocesseds_trainval, high_res_preprocessed_test)

    if high_res:
        source_instance_label = high_res_instance_labels
        source_instance_id = high_res_instance_ids
    else:
        source_instance_label = low_res_instance_labels
        source_instance_id = low_res_instance_ids

    if test:
        scene_ids = scene_ids_test
        if high_res:
            source_ply_path = high_res_extended_ply_files_test
            destination_path = high_res_preprocessed_test
        else:
            source_ply_path = low_res_extended_ply_files_test
            destination_path = low_res_preprocessed_test
    else:
        scene_ids = scene_ids_train + scene_ids_val
        if high_res:
            source_ply_path = high_res_extended_ply_files_trainval
            destination_path = high_res_preprocesseds_trainval
        else:
            source_ply_path = low_res_extended_ply_files_trainval
            destination_path = low_res_preprocessed_trainval

    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)

    if test:
        convert_test_scene_list_file(
            source_ply_path, destination_path, scene_ids, processes)
    else:
        convert_scene_list_id_file(
            source_ply_path, source_instance_id, source_instance_label,
            destination_path, scene_ids, processes)

if __name__ == '__main__':
    #preprocess_id_file_pathes(False, False)
    #preprocess_id_file_pathes(True, False)
    preprocess_id_file_pathes(False, True)
    #preprocess_id_file_pathes(True, True)

# %%