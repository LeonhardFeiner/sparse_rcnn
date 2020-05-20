# %%
import numpy as np

# %%
raw_anchors = np.array([
        [0.3752, 0.3752, 0.4221],
        [0.6566, 0.6566, 0.5159],
        [0.6566, 0.6566, 0.9380],
        [0.4221, 0.4221, 1.6415],
        [0.1876, 1.3132, 1.0318],
        [0.3283, 0.9849, 1.8291],
        [0.7035, 1.5008, 0.8442],
        [1.3132, 0.1876, 1.0318],
        [0.9849, 0.3283, 1.7822],
        [1.5008, 0.7035, 0.8442],
        [0.8442, 2.1574, 0.3752],
        [2.1574, 0.8442, 0.3752],
        [2.4857, 1.1256, 1.0318],
        [1.1256, 2.4857, 1.0318]])

raw_anchor_levels = [raw_anchors[:3], raw_anchors[3:]]
# %%

def get_strides_anchors(anchors, basis_num, min=1, max=None):
    raw_inverse = (basis_num / anchors).round().astype(np.int)
    inverse_clipped = raw_inverse.clip(min=min, max=max)
    calculated_stride_levels, inverse, counts = np.unique(
        inverse_clipped, return_inverse=True, return_counts=True, axis=0)
    order = np.argsort(inverse)
    calculated_anchor_levels = np.split(
        anchors[order], np.cumsum(counts[:-1]), axis=0)
    print(order)
    return calculated_anchor_levels, calculated_stride_levels


calculated_anchors1, calculated_strides1 = get_strides_anchors(
    raw_anchors[:3], 0.6)
calculated_anchors2, calculated_strides2 = get_strides_anchors(
    raw_anchors[3:], 1.2)
calculated_anchor_levels = [calculated_anchors1, calculated_anchors2]
calculated_stride_levels = [calculated_strides1, calculated_strides2]

# %%

all_class_names = np.array([
    '', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
    'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
    'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel',
    'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet',
    'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture',
    'otherprop'])


# %%

keep_segmentation_labels = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
segmentation_label_mapper = np.full(41, -100)
for new_id, old_id in enumerate(keep_segmentation_labels):
    segmentation_label_mapper[old_id] = new_id

num_segmentation_classes = len(keep_segmentation_labels)
segmentation_class_names = all_class_names[keep_segmentation_labels].tolist()

segmentation_class_weights = np.array([
    0.00132814, 0.00175150, 0.00876000, 0.01144720, 0.00573196,
    0.01530075, 0.01158199, 0.00929376, 0.01052470, 0.02300598,
    0.08351005, 0.16326216, 0.01862549, 0.03362379, 0.27182390,
    0.06999986, 0.07244542, 0.12260482, 0.05658397, 0.00879448])

# %%

keep_instance_labels = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
instance_label_mapper = np.full(41, -1)
for new_id, old_id in enumerate(keep_instance_labels):
    instance_label_mapper[old_id] = new_id

instance_label_mapper[[0, 1, 2, 22]] = -1
num_instance_classes = len(keep_instance_labels)
instance_class_names = all_class_names[keep_instance_labels + [-1]].tolist()

instance_class_weights = np.array([
    0.01486793, 0.04012305, 0.00432003, 0.04184261, 0.01557969,
    0.00791617, 0.01785965, 0.05977515, 0.02503404, 0.20921305,
    0.02789507, 0.08875705, 0.19526550, 0.07916169, 0.05138566,
    0.03755106, 0.07510211, 0.00739642, 0.00095407])

# %%
