# %%

import json
import pickle
from functools import partial
from multiprocessing import Pool
import numpy as np
from pathlib import Path

# %%

def get_instance_ids(
        over_segmentation_json, aggregation_json, label_dict):
    seg_indices = over_segmentation_json['segIndices']
    seg_set = set(seg_indices)

    segments_and_newlabels = [
        (agg_group['segments'], label_dict[agg_group['label']])
        for agg_group in aggregation_json['segGroups']]

    filtered_segments_and_newlabels = [
        (segments, label)
        for segments, label
        in segments_and_newlabels
        if label >= 0 and seg_set.intersection(segments)]

    if len(filtered_segments_and_newlabels) == 0:
        return np.full(
            (len(seg_indices),), -1, dtype=int), np.zeros((0,), dtype=int)

    segments_list, labels = zip(*filtered_segments_and_newlabels)

    semantic_labels = np.array(labels)
    aggregation_mapper = np.full(max(seg_indices) + 1, -1)
    for index, segments in enumerate(segments_list):
        aggregation_mapper[segments] = index

    instance_ids = aggregation_mapper[seg_indices]

    return instance_ids, semantic_labels


# %%

def convert_scene_raw(
        label_dict, raw_segs_path, raw_aggregation_path,
        raw_instance_ids_path, raw_instance_labels_path, scene_id):
    print(scene_id)
    segs_path = raw_segs_path.format(scene_id)
    aggregation_path = raw_aggregation_path.format(scene_id)
    instance_ids_path = raw_instance_ids_path.format(scene_id)
    instance_labels_path = raw_instance_labels_path.format(scene_id)

    with open(aggregation_path) as aggregation_file:
        aggregation_json = json.load(aggregation_file)
    with open(segs_path) as segs_file:
        segs_json = json.load(segs_file)

    instance_ids, instance_labels = get_instance_ids(
        segs_json, aggregation_json, label_dict)

    with open(instance_ids_path, 'wb') as instance_ids_file:
        np.save(instance_ids_file, instance_ids)
    with open(instance_labels_path, 'wb') as instance_labels_file:
        np.save(instance_labels_file, instance_labels)


# %%

def convert_scene_list(
        raw_aggregation_path, raw_segs_path, raw_instance_ids_path,
        raw_instance_labels_path, scene_ids, label_map_path, processes=None):

    with open(label_map_path, 'rb') as label_map_file:
        label_dict = pickle.load(label_map_file)

    Path(raw_instance_ids_path).parent.parent.mkdir(exist_ok=True)
    Path(raw_instance_ids_path).parent.mkdir(exist_ok=True)
    Path(raw_instance_labels_path).parent.mkdir(exist_ok=True)

    if processes in (0, False):
        for scene_id in scene_ids:
            convert_scene_raw(
                label_dict, raw_segs_path, raw_aggregation_path,
                raw_instance_ids_path, raw_instance_labels_path, scene_id)
    else:
        convert_scene = partial(
            convert_scene_raw, label_dict, raw_segs_path, raw_aggregation_path,
            raw_instance_ids_path, raw_instance_labels_path)
        p = Pool(processes=processes)
        p.map(convert_scene, scene_ids)
        p.close()
        p.join()


# %%

if __name__ == '__main__':
    import context
    from scannet_config.pathes import (
        label_map_path, aggregation_files_trainval,
        low_res_segs_files_trainval, high_res_segs_files_trainval,
        scene_ids_train, scene_ids_val,
        high_res_instance_ids, low_res_instance_ids,
        high_res_instance_labels, low_res_instance_labels)

    scene_ids = (scene_ids_train + scene_ids_val)
    print('low res')
    convert_scene_list(
        aggregation_files_trainval, low_res_segs_files_trainval,
        low_res_instance_ids, low_res_instance_labels,
        scene_ids, label_map_path)
    print('high res')
    convert_scene_list(
        aggregation_files_trainval, high_res_segs_files_trainval,
        high_res_instance_ids, high_res_instance_labels,
        scene_ids, label_map_path)
# %%