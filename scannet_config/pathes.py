# %%
from pathlib import Path


def get_scene_ids(scene_file_list_filename):
    with open(scene_file_list_filename) as scene_file_list_file:
        return [line.rstrip('\n') for line in scene_file_list_file]


# %%
root_dir = Path('/mnt/raid/leonhard')
# %%

model_storage = (
    root_dir / 'checkpoints' / 'InstanceSegmentation').as_posix() + '/'
log_storage = (
    root_dir / 'tensorboard' / 'InstanceSegmentation').as_posix() + '/'

# %%
data_root_dir = root_dir / 'datasets/ScanNet'
sparse_root_dir = data_root_dir / 'sis_sparse'
sdf_root_dir = data_root_dir / 'sis_sdf'

common_dir = sparse_root_dir / 'common'
origin_dir = sparse_root_dir / 'origin'
labels_dir = sparse_root_dir / 'intermediate'
ply_dir = sparse_root_dir / 'ply'
preprocessed_dir = sparse_root_dir / 'preprocessed'
preaugmented_dir = sparse_root_dir / 'preaugmented'

name_map_path = common_dir / 'label_name_weight_nyu.csv'
label_map_path = common_dir / 'raw_label_dict_nyu.pickle'
scannet_path = common_dir / 'scannetv2-labels.combined.tsv'
ndsis_path = common_dir / 'nyu40labels_scannet.csv'
scene_ids_train_path = common_dir / 'scannetv2_train.txt'
scene_ids_val_path = common_dir / 'scannetv2_val.txt'
scene_ids_test_path = common_dir / 'scannetv2_test.txt'

scene_ids_train = get_scene_ids(scene_ids_train_path)
scene_ids_val = get_scene_ids(scene_ids_val_path)
scene_ids_test = get_scene_ids(scene_ids_test_path)

orig_files_raw = '{1}/{0}/{{0}}/{{0}}{2}'
trainval_dir = 'scans'
test_dir = 'scans_test'

aggregation_files_tuple = 'aggregation', '_vh_clean.aggregation.json'
low_res_ply_files_tuple = 'low_res_ply', '_vh_clean_2.ply'
high_res_ply_files_tuple = 'high_res_ply', '_vh_clean.ply'
low_res_segs_files_tuple = 'low_res_segs', '_vh_clean_2.0.010000.segs.json'
high_res_segs_files_tuple = 'high_res_segs', '_vh_clean.segs.json'

# %%

aggregation_files_trainval = (origin_dir / orig_files_raw.format(
    trainval_dir, *aggregation_files_tuple)).as_posix()
low_res_ply_files_trainval = (origin_dir / orig_files_raw.format(
    trainval_dir, *low_res_ply_files_tuple)).as_posix()
low_res_ply_files_test = (origin_dir / orig_files_raw.format(
    test_dir, *low_res_ply_files_tuple)).as_posix()
high_res_ply_files_trainval = (origin_dir / orig_files_raw.format(
    trainval_dir, *high_res_ply_files_tuple)).as_posix()
high_res_ply_files_test = (origin_dir / orig_files_raw.format(
    test_dir, *high_res_ply_files_tuple)).as_posix()
low_res_segs_files_trainval = (origin_dir / orig_files_raw.format(
    trainval_dir, *low_res_segs_files_tuple)).as_posix()
high_res_segs_files_trainval = (origin_dir / orig_files_raw.format(
    trainval_dir, *high_res_segs_files_tuple)).as_posix()

# %%
ply_file_name = '{}.ply'
low_res_extended_ply_files_trainval = (
    ply_dir / 'low_res_ply_trainval' / ply_file_name).as_posix()
low_res_extended_ply_files_test = (
    ply_dir / 'low_res_ply_test' / ply_file_name).as_posix()
high_res_extended_ply_files_trainval = (
    ply_dir / 'high_res_ply_trainval' / ply_file_name).as_posix()
high_res_extended_ply_files_test = (
    ply_dir / 'high_res_ply_test' / ply_file_name).as_posix()


# %%

instance_id_filename = '{}.npy'
label_id_filename = '{}.npy'

high_res_instance_ids = (
    labels_dir / 'high_res_instance_ids' / instance_id_filename
    ).as_posix()
low_res_instance_ids = (
    labels_dir / 'low_res_instance_ids' / instance_id_filename
    ).as_posix()
high_res_instance_labels = (
    labels_dir / 'high_res_instance_labels' / instance_id_filename
    ).as_posix()
low_res_instance_labels = (
    labels_dir / 'low_res_instance_labels' / instance_id_filename
    ).as_posix()

# %%
torch_file_name = '{}.pth'
low_res_preprocessed_trainval = (
    preprocessed_dir / 'low_res_ply_trainval2' / torch_file_name).as_posix()
low_res_preprocessed_test = (
    preprocessed_dir / 'low_res_ply_test2' / torch_file_name).as_posix()
high_res_preprocesseds_trainval = (
    preprocessed_dir / 'high_res_ply_trainval2' / torch_file_name).as_posix()
high_res_preprocessed_test = (
    preprocessed_dir / 'high_res_ply_test2' / torch_file_name).as_posix()

# %%

sdf_scene_ids_train_path = sdf_root_dir / 'train.txt'
sdf_scene_ids_val_path = sdf_root_dir / 'val.txt'

sdf_scene_ids_train = get_scene_ids(sdf_scene_ids_train_path)
sdf_scene_ids_val = get_scene_ids(sdf_scene_ids_val_path)

chunk_name = '{}.pth'
sdf_chunk_train = (sdf_root_dir / 'train' / chunk_name).as_posix()
sdf_chunk_val = (sdf_root_dir / 'val' / chunk_name).as_posix()

sdf_chunk_sparse_train = (
    sdf_root_dir / 'sparse_train' / chunk_name).as_posix()
sdf_chunk_sparse_val = (
    sdf_root_dir / 'sparse_val' / chunk_name).as_posix()

# %%

preaugmented_path = preaugmented_dir

meshlab_path = (
    'xvfb-run -a -s "-screen 0 800x600x24"'
    ' ~/source/meshlab/src/distrib/meshlabserver')

# %%
evaluation_result_path = root_dir / 'evaluation'

kill_switch_path = '/mnt/raid/killswitch'
kill_switch_person = 'leonhard'
from collections import defaultdict
#%%
