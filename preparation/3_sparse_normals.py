# %%
import os
import context

from preparation.sparse_meshlab import (
    add_normals, point_cloud_selection)

from scannet_config.pathes import (
    scene_ids_train, scene_ids_val, scene_ids_test, meshlab_path,
    low_res_ply_files_trainval, low_res_ply_files_test,
    high_res_ply_files_trainval, high_res_ply_files_test,
    low_res_extended_ply_files_trainval, low_res_extended_ply_files_test,
    high_res_extended_ply_files_trainval, high_res_extended_ply_files_test)

# %%


def add_normal_pathes(high_res=False, test=False):
    if test:
        scene_ids = scene_ids_test
        if high_res:
            source_path = high_res_ply_files_test
            destination_path = high_res_extended_ply_files_test
        else:
            source_path = low_res_ply_files_test
            destination_path = low_res_extended_ply_files_test
    else:
        scene_ids = scene_ids_train + scene_ids_val
        if high_res:
            source_path = high_res_ply_files_trainval
            destination_path = high_res_extended_ply_files_trainval
        else:
            source_path = low_res_ply_files_trainval
            destination_path = low_res_extended_ply_files_trainval

    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)

    add_normals(meshlab_path, source_path, destination_path, scene_ids)


#%%

if __name__ == '__main__':
    #add_normal_pathes(False, False)
    #add_normal_pathes(True, False)
    add_normal_pathes(False, True)
    #add_normal_pathes(True, True)
