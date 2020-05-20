from preparation.prepare_data import raw_aggregation_path
import json
import numpy as np


print('correct', 'scene0217_00')
with open(raw_aggregation_path.format('scene0217_00')) as errorfile:
    faulty_file = json.load(errorfile)

seg_groups = faulty_file['segGroups']

print(len(seg_groups[:31]), len(seg_groups[31:]))
for elem1, elem2 in zip(seg_groups[:31], seg_groups[31:]):
    assert np.array_equal(elem1['segments'], elem2['segments'])

faulty_file['segGroups'] = seg_groups[:31]

with open(raw_aggregation_path.format('scene0217_00'), 'w') as errorfile:
    json.dump(faulty_file, errorfile)