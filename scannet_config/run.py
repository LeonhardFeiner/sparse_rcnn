# %%
import os
import platform
import numpy as np
import torch
import torch.optim as optim
import resource
from collections.abc import Iterable
from itertools import repeat, chain

from ndsis.modules.loss import Loss
from ndsis.training.evaluation import (
    EvaluationHelper, BboxOverlapCalculator, MaskOverlapCalculator,
    ConfusionCalculator, BinaryMaskConfusionCalculator)
from ndsis.utils.determinism import make_deterministic
from ndsis.modules.model import (
    FeatureExtractor, InstanceSegmentationNetwork,
    FeatureLevelDescriptor as FLD)
from scannet_config.network import (
    raw_anchor_levels,
    instance_label_mapper, segmentation_label_mapper,
    num_instance_classes, num_segmentation_classes,
    instance_class_names, segmentation_class_names,
    instance_class_weights, segmentation_class_weights,
    calculated_stride_levels, calculated_anchor_levels)

from scannet_config.pathes import (
    model_storage, log_storage, preaugmented_dir,
    scene_ids_train, scene_ids_val, scene_ids_test,
    low_res_preprocessed_trainval, high_res_preprocesseds_trainval,
    low_res_extended_ply_files_trainval, high_res_extended_ply_files_trainval,
    sdf_scene_ids_train, sdf_scene_ids_val, sdf_chunk_train, sdf_chunk_val)


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

continue_training = None
experiment = None
prefix = ''
suffix = ''
load_state_file = None
loss_load_state_file = None
state_file_dir = model_storage
scene_count_start = 0
scheduler_params = None
data_slice = slice(None)
pre_augmented = False
pre_augmented_epochs = 0
do_eval = True
var_extra_params = None
# %%


# training_name = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_adam'
#     '_lr0004_decay996_step1_weightdecay_batchsize12_bboxsegment'
#     '_mainpathFalse_firstTrue_divisor0_dropinputTrue'
#     '_preaugnyui18s20_mlttsklss_channel32px16_trial2')

# state_file_dir = '/mnt/raid/leonhard/checkpoints/InstanceSegmentationNYU4/'
# load_state_file = training_name +  '/checkpoint/epoch0384.pth'
# loss_load_state_file = training_name + '/loss_checkpoint/epoch0384.pth'

# prefix = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse'
#     '_newstyle_preaugnyui18s20_mlttsklss_channel32px16')


# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse'
#     '_newstyle_preaugnyui18s20_mlttsklss_channel32px16'
#     '_adam_lr0004_decay992_step1_weightdecay_batchsize12_classbboxsegment'
#     '__cs8_dense8max4_reshape_pos32gtTruethres1'
#     '/checkpoint/epoch0512.pth'
# )
# prefix = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse'
#     '_newstyle'#'_preaugnyui18s20_mlttsklss'
#     '_channel32px16'
# #    '_adam_lr0004_decay992_step1_weightdecay_batchsize12_classbboxsegment_'
#     '_cs8_dense8max4_reshape_pos32gtTruethres1'
# )
# experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_channel32px16_cs8_dense8max4_reshape_pos32gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize2x6_maskclassbboxsegment_u32p16m3_rawTrue_unetTrue_skipFalse_pos24neg0gtTruethres2_preaug'
# load_state_file = experiment + '/checkpoint/epoch0064.pth'
# loss_load_state_file = experiment + '/loss_checkpoint/epoch0064.pth'
# continue_training = 512, 614912


# bestunet
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16'
#     '_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12'
#     '_classbboxsegment_mainpathFalse_firstTrue_divisor0_dropinputTrue'
#     '_preaugnyui18s20_loadmlttsk_sndlstft_2downs32c64x128_trial3'
#     '/checkpoint/epoch0336.pth'
# )

# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_32px16_newstyle_preaugnyui18s20_unetmlttsk_2downs32c64x128_adam_lr0004_decay95_step1_weightdecay_batchsize4_maskclassbboxsegment_u32p16m3_rawTrue_unetTrue_skipTrue_pos24neg0gtTruethres1_preaug'
#      '/checkpoint/epoch0008.pth')
# continue_training = 8, 9608

# state_file_dir = '/mnt/raid/leonhard/checkpoints/InstanceSegmentationNYU4/'
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_mainpathFalse_firstTrue_divisor0_dropinputTrue_preaugnyui18s20_loadmlttsk_sndlstft_2downs32c64x128_trial3'
#     #'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_mainpathFalse_firstTrue_divisor0_dropinputTrue_preaugnyui18s20_loadmlttsk_unetclass_2downs32c64x128'
#     '/checkpoint/epoch0632.pth')
# # prefix = (
# #     'inputsize0375_128x128x64_coloronesnormal_sparse_32px16_newstyle_preaugnyui18s20_unetmlttsk_2downs32c64x128'
# # )
# experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit2'

# best bbox_mAP@0.5 
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_doublechannels'
#     '_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox'
#     '_mainpathFalse_firstTrue_divisor0_dropinputTrue'
#     '_preaugnyui18s20_equalweight_nms512_pos32neg8gtTrue'
#     '_secondlastfeat_2downs64c64x128'
#     '/checkpoint/epoch0512.pth')

#----------------------

# dense best bbox
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_dense'
#     '_adam_lr0004_decay996_step1_weightdecay_batchsize62_bbox'
#     '_newstyle_preaugnyui18s20_32px16_dense'
#     '/checkpoint/epoch0384.pth')

# augmented ones only bbox
# load_state_file = (
#     'inputsize0375_128x128x64_ones_sparse_adam'
#     '_lr0004_decay996_step1_weightdecay_batchsize12_bbox_newstyle_32px16'
#     '/checkpoint/epoch0384.pth')

# old anchors small scene bbox
# load_state_file = (
#     'inputsize0469_96x96x48_coloronesnormal_sparse_adam'
#     '_lr0004_decay996_step1_weightdecay_batchsize12_bbox'
#     '_newstyle_liveaugment_32px16_oldanchors128'
#     '/checkpoint/epoch0384.pth')

# sparse allfeat noaugment
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_adam'
#     '_lr0004_decay996_step1_weightdecay_batchsize12_bbox'
#     '_newstyle_liveaugment_32px16_augonlychunck'
#     '/checkpoint/epoch0384.pth')

# best sparse bbox
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_adam'
#     '_lr0004_decay996_step1_weightdecay_batchsize12_bbox'
#     '_newstyle_preaugnyui18s20_32px16'
#     '/checkpoint/epoch0384.pth')

# experiment = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_32px16_newstyle_preaugnyui18s20_unetmlttsk_2downs32c64x128_adam_lr0004_decay95_step1_weightdecay_batchsize4_maskclassbboxsegment_u32p16m3_rawTrue_unetTrue_skipTrue_pos24neg0gtTruethres1_preaug'
#     )
# load_state_file = (
#     experiment +
#      '/checkpoint/epoch0024.pth')
# continue_training = 24, 28824


# load_state_file = (
#     'inputsize0375_128x128x64_ones_sparse_adam'
#     '_lr0004_decay996_step1_weightdecay_batchsize12_bbox'
#     '_newstyle_32px16'
#     '_adam_lr0004_decay996_step1_weightdecay_batchsize12'
#     '_classbbox_32c64x128unit1_pos32neg0gtTruethres1'
#     '/checkpoint/epoch0512.pth')

# prefix = (
#     'inputsize0375_128x128x64_ones_sparse'
#     '_newstyle_32px16'
#     '_32c64x128unit1_pos32neg0gtTruethres1')F


# experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_adam_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_newstyle_liveaugment_32px16_loadtrained_augonlychunckrotatecoordnmirror_32c64x128unit1_pos32neg0gtTruethres1'
# load_state_file = (
#     'inputsize0375_128x128x64_colorones_sparse_adam_lr0004_decay996_step1_weightdecay_batchsize12_bbox_newstyle_32px16_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_32c64x128unit1_pos32neg0gtTruethres1'
# #     'inputsize0375_128x128x64_coloronesnormal_sparse_adam_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_newstyle_liveaugment_32px16_loadtrained_augonlychunckrotate_32c64x128unit1_pos32neg0gtTruethres1'
#     '/checkpoint/epoch0512.pth'
# #     '/checkpoint/epoch0600.pth'
# )
# prefix = 'inputsize0375_128x128x64_colorones_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1'
# #prefix = 'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_augonlychunckrotcoordn_32c64x128unit1_pos32neg0gtTruethres1'
# # # prefix = (
# # #     'inputsize0375_128x128x64_onesnormal_sparse_32px16_newstyle_preaugnyui18s20_unetmlttsk_2downs32c64x128'
# # # )

# load_state_file = (
# #     'inputsize0375_128x128x64_coloronesnormal_sparse_adam_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_newstyle_liveaugment_32px16_loadtrained_augonlychunck_32c64x128unit1_pos32neg0gtTruethres1'
#      'inputsize0375_128x128x64_coloronesnormal_sparse_adam_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_newstyle_liveaugment_32px16_loadtrained_augonlychunckrotatecoordn_32c64x128unit1_pos32neg0gtTruethres1'
#      '/checkpoint/epoch0256.pth'
# )
# prefix = 'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_loadtrained_augonlychunck_32c64x128unit1_pos32neg0gtTruethres1'


##dense----------------------------------------
# load_state_file = (
#      #'inputsize0375_128x128x64_coloronesnormal_dense_adam_lr0004_decay996_step1_weightdecay_batchsize62_bbox_newstyle_preaugnyui18s20_32px16_dense_adam_lr0004_decay996_step1_weightdecay_batchsize4x3_classbbox_32c64x128unit1_16Max8_pos32neg0gtTruethres1'
#      'inputsize0375_128x128x64_coloronesnormal_sparse_adam_lr0004_decay996_step1_weightdecay_batchsize12_bbox_newstyle_preaugnyui18s20_32px16_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_32c64x128unit1_dense16Max8_pos32neg0gtTruethres1'
# #      'inputsize0375_128x128x64_coloronesnormal_sparse_adam_lr0004_decay996_step1_weightdecay_batchsize12_bbox_newstyle_preaugnyui18s20_32px16_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_cs8_dense8max4_reshape_pos32neg0gtTruethres1'
#      '/checkpoint/epoch0512.pth'
# )
# prefix = 'inputsize0375_128x128x64_coloronesnormal_sparse_news_preaugnyui18s20_32px16_32c64x128unit1_dense16Max8_pos32neg0gtTruethres1'
# #prefix = 'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_preaugnyui18s20_32px16_cs8_dense8max4_reshape_pos32neg0gtTruethres1'
# prefix = 'inputsize0375_128x128x64_coloronesnormal_dense_newstyle_preaugnyui18s20_32px16_32c64x128unit1_16Max8_pos32neg0gtTruethres1'
# load_state_file = (
#      'inputsize0469_96x96x48_coloronesnormal_sparse_adam_lr0004_decay996_step1_weightdecay_batchsize12_bbox_newstyle_liveaugment_32px16_oldanchors128_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_32c64x128unit1_pos32neg0gtTruethres1'
# # # #     #'inputsize0375_128x128x64_coloronesnormal_sparse_adam_lr0004_decay996_step1_weightdecay_batchsize12_bbox_newstyle_preaugnyui18s20_32px16_oldanchthres_newres_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_32c64x128unit1_pos32neg0gtTruethres1'
# #      'inputsize0375_128x128x64_coloronesnormal_sparse_adam_lr0004_decay996_step1_weightdecay_batchsize12_bbox_newstyle_preaugnyui18s20_32px16_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_32c64x128unit1_pos32neg0gtTruethres1'
# # # #    '/checkpoint/epoch0640.pth'
#      '/checkpoint/epoch0512.pth'
# )
# # experiment = 'inputsize0469_96x96x48_coloronesnormal_sparse_newstyle_32px16_oldanchors128_32c64x128unit1_pos32neg0gtTruethres1'
# prefix = 'inputsize0469_96x96x48_coloronesnormal_sparse_newstyle_32px16_oldanchors128_32c64x128unit1_pos32neg0gtTruethres1'

# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_adam'
#     '_lr0004_decay996_step1_weightdecay_batchsize12_bbox'
#     '_newstyle_preaugnyui18s20_32px16'
#     '_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox'
#     '_32c64x128unit1_pos32neg0gtTruethres1'
#     '/checkpoint/epoch0512.pth')

# prefix = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16'
#     '_32c64x128unit1_pos32neg0gtTruethres1')

# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_adam_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbbox_newstyle_liveaugment_32px16_loadtrained_augonlychunckrotatecoordnmirror_32c64x128unit1_pos32neg0gtTruethres1_trial2'
#     '/checkpoint/epoch0256.pth'
# )
# prefix = 'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_augchunckrotcoordnmir_32c64x128unit1_pos32neg0gtTruethres1'

# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_trial2'
#     '/checkpoint/epoch0200.pth'
# )
# experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_trainall_lr0001'
# loss_load_state_file = (
#     '../InstanceSegmentationNYU4/'
#     'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_mainpathFalse_firstTrue_divisor0_dropinputTrue_preaugnyui18s20_loadmlttsk_sndlstft_2downs32c64x128_trial3'
#     '/loss_checkpoint/epoch0512.pth')

#experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_preaugnyui18s20_32px16_cs8_dense8max4_reshape_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres2_preaug'
# experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_32px16_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_384_adam_lr0004_decay95_step1_weightdecay_batchsize3x4_maskclassbboxsegment_u32p16m3_rawTrue_unetTrue_skipTrue_pos24neg0gtTruethres2_liveaug'
# load_state_file = experiment + '/checkpoint/epoch0064.pth'
#continue_training = 48, 57648

# load_state_file = (
# # # #     #'inputsize0375_128x128x64_ones_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_liveaug'
# # # #     #'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_augonlychunck_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_liveaug'
# # # #     #'inputsize0375_128x128x64_onesnormal_sparse_32px16_newstyle_preaugnyui18s20_unetmlttsk_2downs32c64x128_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_liveaug'
# # # #     #'inputsize0375_128x128x64_ones_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_liveaug'
# # # #     #'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_preaug'
# #  #   'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres2_preaug'
# # # #     'inputsize0375_128x128x64_colorones_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_liveaug'
# # # #     'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres3_preaug'
# # #     'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipFalse_pos24neg0gtTruethres1_preaug'
# #     #'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos32neg0gtFalsethres1_liveaug'
# #     #'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_augchunckrotcoordnmir_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_liveaug'
# #     #'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_augchunckrotcoordnmirr_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres1_liveaug'
#     'inputsize0375_128x128x64_coloronesnormal_sparse_news_preaugnyui18s20_32px16_32c64x128unit1_dense16Max8_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipFalse_pos24neg0gtTruethres2_preaug'
#     '/checkpoint/epoch0064.pth'
# )

# experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_newstyle_32px16_32c64x128unit1_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize6x2_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipTrue_pos24neg0gtTruethres2_preaug'
# continue_training = 64, 76864

#experiment = 'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_trainall_lr0001'
#load_state_file = (experiment + '/checkpoint/epoch0256.pth')
# loss_load_state_file = (experiment + '/loss_checkpoint/epoch0256.pth')
# continue_training =  256, 307456

# prefix = 'inputsize0375_128x128x64_coloronesnormal_sparse_32px16_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_384'
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_trainall_lr0001'
#     '/checkpoint/epoch0384.pth')

#experiment = 'inputsize0375_128x128x64_coloronesnormal_dense_newstyle_preaugnyui18s20_32px16_32c64x128unit1_16Max8_pos32neg0gtTruethres1_adam_lr0004_decay95_step1_weightdecay_batchsize4x3_maskclassbbox_u32p16m3_rawTrue_unetFalse_skipFalse_pos24neg0gtTruethres2_preaug'
#continue_training =  32, 38432
#load_state_file = (experiment + '/checkpoint/epoch0064.pth')


# prefix = 'inputsize0375_128x128x64_coloronesnormal_sparse_32px16_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_384'
# load_state_file = (
#     'inputsize0375_128x128x64_coloronesnormal_sparse_channel32px16_bboxsegment_adam_lr0004_decay996_step1_weightdecay_batchsize12_classbboxsegment_newstyle_preaugnyui18s20_loadmlttsk_sndlstft_32c64x128unit1_trainall_lr0001'
#     '/checkpoint/epoch0384.pth')

# var_extra_params = dict(
#     theta=None,
#     sub_pixel_offset=None,
#     max_empty_border_size_divisor=None,
#     mirror=None,
#     coord_noise_sigma=0.1,
#     color_noise_sigma=0,
#     normal_noise_sigma=0,
#     common_color_noise=False,
#     common_normal_noise=False,
#     shift=None,
#     shuffle=False,)

suffix = 'newstyle_32px16_nomultitask'
#suffix = ('_cs8_dense8max4_reshape_pos32gtTruethres1')

gpu = None
calc_bbox = True
calc_class = False
calc_mask = False
calc_segmentation = False
include_unet = calc_segmentation
debug = True
detail = False
sdf = False
sparse = False
input_scale_neg_exponent = 2  # values 2 and 3 make sense
use_color = True
use_ones = True
use_normal = True
determinism = None
ignore_load = True
load_shape_filter = False
unet_tweak = False
load_rpn_only = True
load_class_only = calc_mask
mask_only = calc_mask
overfit = False
load_using_gpu = False
use_wandb = False
sgd = False
preload = True
num_workers = 0 if debug else (4 if preload else 6)
upconvoluted_anchornetwork = True
dense_class = True
simple_class = False #dense_class
# pre_augmented = (
#     'voxelsize0375_NYUi18s20' if 
#     (
#         use_color and use_normal and use_ones
#         and var_extra_params is None
#         and upconvoluted_anchornetwork
#     )
#     else False)
pre_augmented_epochs = 256 if pre_augmented else 0

train_batch_size = 12
val_batch_size = 6
batches_per_step = 1  # ([1] * 128) + ([2] * 128) + ([3] * 128)


small_last_layer_shapes = [
    (12, 12, 6), (12, 12, 8), (16, 16, 8), (20, 20, 12), (24, 24, 12), (32, 32, 16)]
last_layer_val_scene_size = small_last_layer_shapes[-1]

if upconvoluted_anchornetwork:
    input_voxel_size = 0.0375
    positive_overlap = 0.40
    last_layer_train_scene_size = small_last_layer_shapes[2]
    negative_overlap = 0.25
else:
    positive_overlap = 0.35
    negative_overlap = 0.15
    input_voxel_size = 0.0469
    last_layer_train_scene_size = small_last_layer_shapes[0]
    pre_augmented = False

if pre_augmented:
    assert input_voxel_size == 0.0375
    assert input_scale_neg_exponent == 2

if calc_mask:
    batches_per_step = 2
    train_batch_size = 6
    batches_per_step = 2
    val_batch_size = 4
    if calc_segmentation or not sparse:
        # batches_per_step = 4
        # train_batch_size = 3
        batches_per_step = 6
        train_batch_size = 2
        val_batch_size = 2


if not sparse:
    train_batch_size = 6
    batches_per_step = 2
    val_batch_size = 1
    if calc_class:
        train_batch_size = 4
        batches_per_step = 3

mask_positive_threshold = 0.2
positive_threshold = 0.1

background_label = -100

optim_params = dict(
    #lr=5e-5,
#    lr=1e-4,
    #lr=2e-4,
    lr=4e-4,
#    lr=8e-4
#    lr=16e-4,
    #lr=32e-4,
    #lr=64e-4,
    #lr=128e-4,
    #lr=256e-4,
    #weight_decay=0.01,
    weight_decay=0,
    )
#scheduler_params = dict(step_size=1, gamma=0.996)  # 1/5
scheduler_params = dict(step_size=1, gamma=0.992)  # 1/22
# scheduler_params = dict(step_size=1, gamma=0.988)  # 1/100

if debug:
    #num_workers = 4
    data_slice = slice(12 * 2)
    #preload = False
    #experiment = 'debug'
    #overfit = 8
    #preload = False
    #train_batch_size = 2
    #val_batch_size = 2
    # epochs = range(1)
    # do_eval = False
    pass
if sdf:
    num_anchor_pathes = 2
    input_scale_neg_exponent = 2  # 2 ###

    evaluate_every_step = 512
    evaluate_every_epoch = 0
    epochs = range(0, 7)
    shuffle = False
    if not sparse:
        train_batch_size = 16
        val_batch_size = 32
        input_channels = 2
        use_color = False
        use_ones = False
        use_normal = False
    else:
        use_color = False
        use_ones = True
        use_normal = False
        input_channels = 1

else:
    num_anchor_pathes = 1
    evaluate_every_step = 0
    evaluate_every_epoch = 8
    epochs = range(0, 256 + 128)
    if calc_class:
        epochs = range(0, 512 + 256)
    input_channels = 0
    if use_color:
        input_channels += 3
    if use_ones:
        input_channels += 1
    if use_normal:
        input_channels += 3

if mask_only:
    if sgd:
        optim_params['lr'] = 0.04
    epochs = range(64)
    scheduler_params = dict(step_size=1, gamma=0.95)

if overfit:
    pre_augmented = False
    pre_augmented_epochs = 0
    batches_per_step = 1
    epochs = range(0, 1024)
    evaluate_every_epoch = 16
    if scheduler_params is not None:
        scheduler_params = dict(step_size=256, gamma=0.5)


scale_factor = 2 ** (input_scale_neg_exponent + 1)
last_layer_voxel_size = input_voxel_size * scale_factor
train_scene_size = tuple(
    size * scale_factor for size in last_layer_train_scene_size)
val_scene_size = tuple(
    size * scale_factor for size in last_layer_val_scene_size)

print(train_scene_size, val_scene_size)

print(
    tuple(input_voxel_size * np.array(train_scene_size)),
    tuple(input_voxel_size * np.array(val_scene_size)))

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if gpu is not None else ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not debug:
    assert torch.cuda.is_available()

if determinism is not None:
    make_deterministic(determinism)

if continue_training is not None:
    load_rpn_only = False
    load_class_only = False
    start_epoch, scene_count_start = continue_training
    scene_count_start -= train_batch_size
    continue_training = start_epoch, scene_count_start
    epochs = range(start_epoch, max(epochs) + 1)
# %%

main_path_relu = False
relu_first = True
bottleneck_divisor = 0
drop_input_relu = True
num_positive_boxes_class = 32
use_gt_boxes_class = True


num_anchor_pathes = 1
anchor_output_channels =  128
bottleneck_dilation = False
extra_stride_levels = None

smaller_layers = 0
anchor_size_offset = 0

if upconvoluted_anchornetwork:
    extra_stride_levels = calculated_stride_levels
    raw_anchor_levels = calculated_anchor_levels
    num_anchor_pathes = len(calculated_stride_levels)
    smaller_layers = 1
    anchor_output_channels = [128, 256]

minimal_channels = 32
additional_channels = 16

selected_channels_definition = (
    np.arange(1 + input_scale_neg_exponent + smaller_layers)
    * additional_channels + minimal_channels)


start_channels = selected_channels_definition[0]
in_between_channels = selected_channels_definition[1:-num_anchor_pathes]
anchor_channels = selected_channels_definition[-num_anchor_pathes:]

# %%
anchor_dict = dict()

in_between_downsampler = (
    FLD('B', channels)
    for channels in in_between_channels)

method = 'D'
if bottleneck_dilation:
    anchor_dict = dict(
        make_dense=True,
        stride=1,
        num_bottlenecks=anchor_dict['num_dilations'])
    method = 'B'

if not isinstance(anchor_output_channels, Iterable):
    anchor_output_channels = repeat(anchor_output_channels)

anchor_descriptions = (
    [FLD(method, anchor_output_channels_single, anchor_dict)]
    for anchor_output_channels_single in anchor_output_channels)

if not calc_bbox:
    anchor_descriptions = [None] * len(anchor_output_channels)

anchor_outputs = (
    FLD('B', channels, dict(), anchor_description)
    for channels, anchor_description
    in zip(anchor_channels, anchor_descriptions))

layer_descriptions = [
    FLD('B', start_channels, dict(stride=1, drop_input_relu=True)),
    *in_between_downsampler,
    *anchor_outputs, ]


if include_unet:
    layer_descriptions.append(
        FLD('B', max(selected_channels_definition) + additional_channels * 1))
    layer_descriptions.append(
        FLD('B', max(selected_channels_definition) + additional_channels * 2))

print(layer_descriptions)

if not sparse and not sdf:
    layer_descriptions = [FLD('C', input_channels)] + layer_descriptions


sparse_input = (not sdf or sparse)
feature_extractor_params = dict(
    num_dims=3, sparse=sparse_input, input_channels=input_channels,
    network_description=layer_descriptions,
    class_output_anchor=dense_class,
    class_output_upsampled=False,
    class_output_index=(
        (0 if dense_class else -4) 
        if calc_segmentation else
        (-2 if upconvoluted_anchornetwork else -1)),
    num_dilations=5, num_units=2,
    bottleneck_divisor=bottleneck_divisor, stride=2, maxpool=False,
    relu_first=relu_first,
    main_path_relu=main_path_relu, bottleneck_groups=1, batchnorm=False,
    use_residuals=True, drop_input_relu=drop_input_relu,)
unet_params = dict(
    use_residuals=feature_extractor_params['use_residuals'],
    num_units=feature_extractor_params['num_units'],
    bottleneck_divisor=feature_extractor_params['bottleneck_divisor'],
    groups=feature_extractor_params['bottleneck_groups'],
    main_path_relu=feature_extractor_params['main_path_relu'],
    relu_first=feature_extractor_params['relu_first'],
    batchnorm=feature_extractor_params['batchnorm'],
    concat=True,
    min_channels=16)

if include_unet:
    feature_extractor_params['include_unet'] = True
    feature_extractor_params['unet_params'] = unet_params

if num_anchor_pathes == 1 and not upconvoluted_anchornetwork:
    raw_anchor_levels = [np.concatenate(raw_anchor_levels)]

# %%
pooling_function_or_none = torch.mean  # or None

class_input_network_description = class_input_network_description = [FLD('I')]
if not sparse or feature_extractor_params['class_output_anchor']:
    pooling_function_or_none = pooling_function_or_none
    cut_shape = (16, 16, 16)
    if pooling_function_or_none is None:
        class_output_network_description = [FLD('S', None), FLD('S', 8)]
    else:
        class_output_network_description = [FLD('S', 265), FLD('S', 512)]
else:
    pooling_function_or_none = torch.mean
    cut_shape = None
    class_output_network_description = [
        FLD('S', None),
        FLD('S', None),
        ]

common_class_dict = dict(
    main_path_relu=main_path_relu,
    relu_first=relu_first,
    bottleneck_divisor=bottleneck_divisor,
    drop_input_relu=drop_input_relu,
    make_dense=False,
    num_units=1)

class_input_network_description = [
    # FLD(
    #     type='B',
    #     channels=16,
    #     params={**common_class_dict, 'stride': 1},
    #     anchor_path=None),
    FLD(
        type='B',
        channels=32,
        params={**common_class_dict, 'stride': 1},
        anchor_path=None),
    # FLD(
    #     type='B',
    #     channels=64,
    #     params={**common_class_dict, 'stride': 1},
    #     anchor_path=None),
    # FLD(
    #     type='B',
    #     channels=128,
    #     params={**common_class_dict, 'stride': 1},
    #     anchor_path=None),
    ]
class_output_network_description = [
    # FLD(
    #     type='B',
    #     channels=32,
    #     params={**common_class_dict, 'stride': 2},
    #     anchor_path=None),
    FLD(
        type='B',
        channels=64,
        params={**common_class_dict, 'stride': 2},
        anchor_path=None),
    FLD(
        type='B',
        channels=128,
        params={**common_class_dict, 'stride': 2},
        anchor_path=None),
    # FLD(
    #     type='B',
    #     channels=256,
    #     params={**common_class_dict, 'num_units': 2, 'stride': 2},
    #     anchor_path=None),
    ]

pooling_function_or_none = torch.mean

if dense_class and simple_class:
    class_input_network_description = [FLD('I')]
    cut_shape = (8, 8, 8)
    class_output_network_description = [FLD('S', 8)]
    pooling_function_or_none = None
elif False:
    class_output_network_description.pop()
    class_output_network_description.append(FLD('S', 8))
    pooling_function_or_none = None

if not sparse or feature_extractor_params['class_output_anchor']:
    class_output_network_description = [
        FLD('M', params={'stride': 2}), *class_output_network_description]

class_network_params = dict(
    input_network_description=class_input_network_description,
    output_network_description=class_output_network_description,
    linear_channels=[64],
    num_classes=num_instance_classes,
    raw_scene=False,
    cut_shape=cut_shape,
    pooling_function_or_none=pooling_function_or_none,
    relu_after_pooling=True,
    positive_threshold=positive_threshold,
    negative_threshold=0,
    selection_tuple=(
        num_positive_boxes_class, 0, use_gt_boxes_class))

semantic_network_params = dict(
    num_classes=num_segmentation_classes,
)

# %%


common_mask_dict = dict(
    use_residuals=True,
    main_path_relu=main_path_relu,
    relu_first=relu_first,
    bottleneck_divisor=bottleneck_divisor,
    drop_input_relu=drop_input_relu,
    make_dense=False)

mask_input_network_description = [
    FLD(
        type='B',
        channels=16,
        params={**common_mask_dict, 'num_units': 2},
        anchor_path=None),
    ]
mask_unet = True
if mask_unet:
    mask_unet_params = unet_params.copy()
    mask_output_network_description = [
        FLD('I'),
        FLD(
            type='B',
            channels=32,
            params={**common_mask_dict, 'num_units': 2, 'stride': 2},
            anchor_path=None),
        FLD(
            type='B',
            channels=48,
            params={**common_mask_dict, 'num_units': 2, 'stride': 2},
            anchor_path=None),
        FLD(
            type='B',
            channels=64,
            params={**common_mask_dict, 'num_units': 2, 'stride': 2},
            anchor_path=None),
        # FLD(
        #     type='B',
        #     channels=80,
        #     paframs={**common_mask_dict, 'num_units': 2, 'stride': 2},
        #     anchor_path=None),
        # FLD(
        #     type='B',
        #     channels=96,
        #     params={**common_mask_dict, 'num_units': 2, 'stride': 2},
        #     anchor_path=None),
    ]

else:
    mask_unet_params= dict()
    mask_output_network_description = [
        FLD(
            type='B',
            channels=16,
            params={**common_mask_dict, 'num_units': 4},
            anchor_path=None)
    ]


mask_network_params = dict(
    use_raw_features=True,
    use_unet_features=calc_segmentation,
    use_skip_features=False,
    internal_unet=mask_unet,
    unet_params=mask_unet_params,
    input_network_description=mask_input_network_description,
    output_network_description=mask_output_network_description,
    channel_list=[32, num_instance_classes],
    positive_threshold=mask_positive_threshold,
    selection_tuple=(24, 0, True)
)

if calc_mask:
    selection_tuple = mask_network_params['selection_tuple']
    thresholdstr = str(mask_network_params["positive_threshold"])[2:]
    suffix = (
        f'{"u32p16m3" if mask_unet else "16m4"}'
        f'_raw{mask_network_params["use_raw_features"]}'
        f'_unet{mask_network_params["use_unet_features"]}'
        f'_skip{mask_network_params["use_skip_features"]}'
        f'_pos{selection_tuple[0]}neg{selection_tuple[1]}gt{selection_tuple[2]}'
        f'thres{thresholdstr}'
        f'_{"preaug" if pre_augmented else "liveaug"}'
    )

# %%

if sdf:
    input_voxel_size = 0.0469
    
if upconvoluted_anchornetwork:
    resized_anchor_levels = [
        [raw_anchors / input_voxel_size for raw_anchors in raw_anchor_level]
        for raw_anchor_level in raw_anchor_levels]
else:
    resized_anchor_levels = [
        raw_anchor_level / input_voxel_size
        for raw_anchor_level in raw_anchor_levels]

anchor_network_params = dict(
    raw_anchor_levels=resized_anchor_levels,
    allowed_border=0,
    store_anchors=True,
    extra_stride_levels=extra_stride_levels,
    store_goal_device=True,
)

roi_selector_params = dict(
    num_keep_pre_nms=1024,
    num_keep_post_nms=256,
    thresh_nms=0.5,
    val_num_keep_pre_nms=1024,
    val_num_keep_post_nms=32 if calc_mask else 256,
    val_thresh_nms=0.3,
)

# %%

if False:
    mask_class_weights = torch.as_tensor(
        instance_class_weights, dtype=torch.get_default_dtype())
else:
    mask_class_weights = None

if False:
    class_class_weights = torch.as_tensor(
        instance_class_weights, dtype=torch.get_default_dtype())
else:
    class_class_weights = None

if False:
    segmentation_weights = torch.as_tensor(
        segmentation_class_weights, dtype=torch.get_default_dtype())
else:
    segmentation_weights = None

loss_params = dict(
    multitask_loss=False,
    batchwise_subsample=True,
    positive_overlap=positive_overlap,
    negative_overlap=negative_overlap,
    max_weight=1/8,  # 1 is default
    mask_class_weights=mask_class_weights,
    class_class_weights=class_class_weights,
    sigma=2.,
    sparse=sparse_input,
    mask_positive_threshold=mask_network_params.get('positive_threshold'),
    class_positive_threshold=class_network_params.get('positive_threshold'),
    class_negative_threshold=0,
    class_negative_label=-100,
    calc_mask=calc_mask,
    calc_class=calc_class and not mask_only,
    calc_segmentation=calc_segmentation and not mask_only,
    calc_bbox=calc_bbox and not mask_only,
    segmentation_weights=segmentation_weights,
    loss_rpn_bbox_weight=1,
    loss_rpn_class_weight=1,
    # loss_segmentation_weight=1/8,
)

model_params = dict(
    feature_extractor_params=feature_extractor_params,
    class_network_params=class_network_params if calc_class else dict(),
    mask_network_params=mask_network_params if calc_mask else dict(),
    segmentation_params=(
        semantic_network_params if calc_segmentation else dict()),
    anchor_network_params=anchor_network_params,
    roi_selector_params=roi_selector_params,
    include_bbox_network=calc_bbox,
    include_class_network=calc_class,
    include_mask_network=calc_mask,
    include_segmentation_network=calc_segmentation,
    )


# %%

score_threshold = 0.9
mask_threshold=0.5
evaluation_helper = EvaluationHelper(
    overlap_thresholds=[
        0.25, 0.5,  # ('[0.5:0.95:0.05]', np.arange(50, 100, 5) / 100)
    ],
    overlap_class_indices=range(18), 
    overlap_class_names=instance_class_names,
    overlap_methods=[None],
    confusion_class_names=segmentation_class_names,
    device=device)

bbox_overlap_calculator = BboxOverlapCalculator(score_threshold)
mask_overlap_calculator = MaskOverlapCalculator(
    mask_threshold=mask_threshold, score_threshold=score_threshold)
confusion_calculator = ConfusionCalculator(num_segmentation_classes)
label_confusion_calculator = ConfusionCalculator(num_instance_classes)
mask_confusion_calculator = BinaryMaskConfusionCalculator(mask_threshold)

# %%

orthogonal_angles = np.arange(4) / 2 * np.pi

if not sdf:
    common_params = dict(
        instance_cutoff_threshold=0.8,
        additional_bbox_pixel=0,
        scale=(1 / input_voxel_size),
        background_label=background_label,
        instance_label_mapper=torch.tensor(instance_label_mapper),
        segmentation_label_mapper=torch.tensor(segmentation_label_mapper),
        use_color=use_color,
        use_ones=use_ones,
        use_normal=use_normal,
        device=(device if load_using_gpu else None),
        collate_augmentation=load_using_gpu,
        preload=preload,
        pre_augmented=pre_augmented,
        data_slice=data_slice)

    fix_params = dict(
        common_params,
        theta=0,
        sub_pixel_offset=0,
        max_empty_border_size_divisor=None,
        mirror=False,
        coord_noise_sigma=0,
        color_noise_sigma=0,
        normal_noise_sigma=0,
        common_color_noise=False,
        common_normal_noise=False,
        shift=0,
        shuffle=False,)

    if var_extra_params is None:
        var_params = dict(
            common_params,
            theta=None,
            sub_pixel_offset=None,
            max_empty_border_size_divisor=None,
            mirror=None,
            coord_noise_sigma=0.1,
            color_noise_sigma=0.1,
            normal_noise_sigma=0,
            common_color_noise=False,
            common_normal_noise=False,
            shift=None,
            shuffle=True,)
    else:
        var_params = {**common_params, **var_extra_params}

    from ndsis.data.data import get_data_loader, batch_to_device
    overfit_loader_params = dict(
        fix_params, batch_size=train_batch_size, spatial_size=train_scene_size,
        data_slice=slice(0, overfit), num_workers=num_workers)
    train_loader_params = dict(
        var_params, batch_size=train_batch_size, spatial_size=train_scene_size,
            num_workers=num_workers,)
    val_loader_params = dict(
        fix_params, batch_size=val_batch_size, spatial_size=val_scene_size,
            num_workers=num_workers // 2, shift=8
        )

    if overfit:
        print(overfit_loader_params)
        print(val_loader_params)
        train_loader_params = overfit_loader_params
        val_loader_params = overfit_loader_params


    # assert not pre_augmented, pre_augmented
    assert isinstance(detail, bool), detail

    selected_train_scene_ids = scene_ids_train
    selected_val_scene_ids = scene_ids_val

    if pre_augmented:
        selected_train_path = str(
            preaugmented_dir / pre_augmented / 'train' / 'epoch{:0>3}' /
            '{{}}.pth')
        selected_val_path = str(
            preaugmented_dir / pre_augmented / 'val' / '{}.pth')
        selected_trainval_path = str(
            preaugmented_dir / pre_augmented / 'trainval' / '{}.pth')
        train_loader_params['epochs'] = pre_augmented_epochs
        val_loader_params['epochs'] = 0
        selected_ply_path = low_res_extended_ply_files_trainval
    elif detail:
        selected_train_path = high_res_preprocesseds_trainval
        selected_val_path = high_res_preprocesseds_trainval
        selected_ply_path = high_res_extended_ply_files_trainval
        selected_trainval_path = selected_train_path
    else:
        selected_train_path = low_res_preprocessed_trainval
        selected_val_path = low_res_preprocessed_trainval
        selected_ply_path = low_res_extended_ply_files_trainval
        selected_trainval_path = selected_train_path

else:
    from ndsis.data.sdf import (
        get_data_loader, batch_to_device, batch_to_device_sparse)

    if sparse:
        batch_to_device = batch_to_device_sparse

    common_params = dict(dense=not sparse, num_workers=num_workers)
    if overfit:
        train_loader_params = dict(
            **common_params, batch_size=train_batch_size,
            data_slice=slice(0, overfit), shuffle=shuffle)
        val_loader_params = dict(
            **common_params, batch_size=val_batch_size,
            data_slice=slice(0, overfit), shuffle=False)
    else:
        train_loader_params = dict(
            **common_params, batch_size=train_batch_size, shuffle=shuffle)
        val_loader_params = dict(
            **common_params, batch_size=val_batch_size, shuffle=False)

    selected_train_scene_ids = sdf_scene_ids_train
    selected_val_scene_ids = sdf_scene_ids_val

    selected_train_path = sdf_chunk_train
    selected_val_path = sdf_chunk_val
    selected_trainval_path = selected_train_path
    selected_ply_path = low_res_extended_ply_files_trainval

# %%

if load_state_file and not prefix and not continue_training:
    prefix = load_state_file.split('/')[0]

model = InstanceSegmentationNetwork(**model_params)
loss = Loss(**loss_params)
trainable_parameter_count = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
# %%

param_configs = {
    'train_loader': train_loader_params,
    'val_loader': val_loader_params,
    'model': model_params,
    'feature_extractor': feature_extractor_params,
    'class_network': class_network_params if calc_class else {},
    'mask_network': mask_network_params if calc_mask else {},
    'segmentation_network': (
        semantic_network_params if calc_segmentation else dict()),
    'loss': loss_params,
    'optim': optim_params,
    'roi_selector': roi_selector_params,
    'anchor_network': anchor_network_params,
    'scheduler': {} if scheduler_params is None else scheduler_params,
}

remaining_config = {
    'computername': platform.node(),
    'gpu': gpu,
    'calc_bbox': calc_bbox,
    'calc_class': calc_class,
    'calc_mask': calc_mask,
    'calc_segmentation': calc_segmentation,
    'detail': detail,
    'input_scale_neg_exponent': input_scale_neg_exponent,
    'use_color': use_color,
    'use_ones': use_ones,
    'use_normal': use_normal,
    'mask_only': mask_only,
    'load_using_gpu': load_using_gpu,
    'num_workers': num_workers,
    'sgd': sgd,
    'preload': preload,
    'pre_augmented': pre_augmented,
    'device': str(device),
    'debug': debug,
    'sdf': sdf,
    'sparse': sparse,
    'scheduler': scheduler_params,
    'evaluate_every_step': evaluate_every_step,
    'evaluate_every_epoch': evaluate_every_epoch,
    'epochs': epochs,
    'determinism': determinism,
    'load_state_file': load_state_file,
    'load_rpn_only': load_rpn_only,
    'load_class_only': load_class_only,
    'overfit': overfit,
    'batches_per_step': batches_per_step,
    'last_layer_voxel_size': last_layer_voxel_size,
    'trainable_parameter_count': trainable_parameter_count,
}

complete_config = {**param_configs, **remaining_config}

if isinstance(batches_per_step, int):
    batches_per_step_string = batches_per_step_string_raw = (
        '' if batches_per_step <= 1 else f'x{batches_per_step}')
else:
    batches_per_step_string_raw = '-'.join(
        f'{x}x{y}' for x,y in batches_per_step)
    batches_per_step = np.concatenate([
        [x] * y for x, y in batches_per_step
    ])
    batches_per_step_string = 'xsteps-' + batches_per_step_string_raw

type_name = (
    ('mask' if calc_mask else '') +
    ('class' if calc_class else '') +
    ('bbox' if calc_bbox else '') +
    ('segment' if calc_segmentation else ''))

if sdf:
    feat_name = 'sdf'
elif not any((use_color, use_ones, use_normal)):
    feat_name = 'empty'
else:
    feat_name = ''
    if use_color:
        feat_name += 'color'
    if use_ones:
        feat_name += 'ones'
    if use_normal:
        feat_name += 'normal'

size_name = ((
    f'inputsize{str(input_voxel_size)[2:]}_'
    f'{"x". join(str(dim_size) for dim_size in train_scene_size)}_'
    f'{feat_name}_'
    f'{"sparse" if sparse else "dense"}')
    if not prefix else
    prefix)
schedulername = "" if scheduler_params is None else (
    f'decay{str(scheduler_params["gamma"])[2:]}_'
    f'step{scheduler_params["step_size"]}_')

overfit_or_batchsize = (
    f'overfit{overfit}'
    if overfit else
    (
        f'batchsize{train_loader_params["batch_size"]}'
        f'{batches_per_step_string}'))

wd_string = f'{optim_params["weight_decay"]:f}'[2:].rstrip('0')
lr_string = f'{optim_params["lr"]:f}'[2:].rstrip('0')
experiment_string = (
    f'{size_name}_{"sgd" if sgd else "adam"}_'
    f'lr{lr_string}_{schedulername}'
    f'weightdecay{wd_string}_'
    f'{overfit_or_batchsize}_{type_name}'
    f'{"_" if suffix else ""}{suffix}')

if experiment is None:
    experiment = experiment_string
else:
    print('Experiment name would be', experiment_string)
print('strlen', len(experiment))

# %%
flat_config = {
    'description': suffix,
    'debug': debug,
    'sparse': sparse,
    'gpu': -1 if gpu is None else gpu,
    'computername': platform.node(),
    'calc_class': calc_class,
    'calc_mask': calc_mask,
    'mask_only': mask_only,
    'determinism': bool(determinism),
    'determinism_value': 0 if determinism is None else int(determinism),
    'load_state_file': '' if load_state_file is None else load_state_file,
    'load_rpn_only': load_rpn_only,
    'load_class_only': load_class_only,
    'overfit': overfit if overfit else -1,
    'batches_per_step': batches_per_step_string,
    'last_layer_voxel_size': last_layer_voxel_size,
    'epochs': max(epochs),
    'optim/sgd': sgd,
    'optim/lr': optim_params['lr'],
    'optim/weight_decay': optim_params.get('weight_decay', 0.),
    'optim/momentum': optim_params.get('momentum', 0.),
    'scheduler/used': bool(scheduler_params),
    'scheduler/step_size': (
        scheduler_params.get('step_size', 1) if scheduler_params else 0),
    'scheduler/gamma': (
        scheduler_params.get('gamma', 1) if scheduler_params else 0),
    'batch_size': train_batch_size,
    'data/instance_cutoff_threshold': train_loader_params.get(
        'instance_cutoff_threshold', 0.8),
    'device': device,
    'num_workers': num_workers,
    'data/input_scale_neg_exponent': input_scale_neg_exponent,
    'data/input_voxel_size': input_voxel_size,
    'data/sdf': sdf,
    'data/color': use_color,
    'data/use_ones': use_ones,
    'data/use_normal': use_normal,
    'data/pre_aug': bool(
        train_loader_params.get('pre_augmented', False)),
    'data/rand_theta': train_loader_params.get('theta', False) is None,
    'data/rand_theta_value': (
        train_loader_params.get('theta', 0)
        if isinstance(train_loader_params.get('theta', 0), int) else 0),
    'data/rand_theta_values': (
        str(train_loader_params.get('theta', None))),
    'data/rand_mirror': train_loader_params.get('mirror', False) is None,
    'data/rand_mirror_value': (
        False if train_loader_params.get('mirror', False) is None else
        train_loader_params.get('mirror', False)),
    'data/rand_sub_pixel_offset': train_loader_params.get(
        'sub_pixel_offset', False) is None,
    'data/rand_sub_pixel_offset_value': (
        0 if train_loader_params.get('sub_pixel_offset', 0) is None else
        train_loader_params.get('sub_pixel_offset', 0)),
    'data/rand_shift': train_loader_params.get('shift', None) is None,
    'data/rand_shift_value': (
        0 if train_loader_params.get('shift', 0) is None else
        train_loader_params.get('shift', 0)),
    'data/empty_border': train_loader_params.get(
        'max_empty_border_size_divisor', None) is not None,
    'data/empty_border': (
        0
        if train_loader_params.get('max_empty_border_size_divisor', 0) is None
        else train_loader_params.get('max_empty_border_size_divisor', 0)),
    'data/shuffle': train_loader_params.get('shuffle', False),
    'data/coord_noise_sigma': train_loader_params.get(
        'coord_noise_sigma', 0),
    'data/color_noise_sigma': train_loader_params.get(
        'color_noise_sigma', 0),
    'data/normal_noise_sigma': train_loader_params.get(
        'normal_noise_sigma', 0),
    'data/common_color_noise': train_loader_params.get(
        'common_color_noise', True),
    'data/common_normal_noise': train_loader_params.get(
        'common_normal_noise', True),
    'data/detail': detail,
    'data/load_using_gpu': load_using_gpu,
    'data/preload': preload,
    'data/num_workers': num_workers,
    'feature_extractor/layer_description': str(layer_descriptions),
    'feature_extractor/bottleneck_divisor':
        feature_extractor_params['bottleneck_divisor'],
    'feature_extractor/maxpool':
        feature_extractor_params['maxpool'],
    'feature_extractor/batchnorm':
        feature_extractor_params['batchnorm'],
    'feature_extractor/num_units':
        feature_extractor_params['num_units'],
    'feature_extractor/stride':
        str(feature_extractor_params['stride']),
    'feature_extractor/num_dilations':
        feature_extractor_params['num_dilations'],
    'feature_extractor/main_path_relu':
        feature_extractor_params['main_path_relu'],
    'class_network/sparse': bool(model.feature_extractor.class_sparse),
    'class_network/align': isinstance(cut_shape, tuple),
    'class_network/cut_shape': str(cut_shape),
    'class_network/globalpool': bool(pooling_function_or_none is not None),
    'class_network/linear_channels':
        str(class_network_params['linear_channels']),
    'class_network/relu_after_pooling':
        class_network_params['relu_after_pooling'],
    'class_network/class_input_network_description':
        str(class_input_network_description),
    'class_network/class_output_network_description':
        str(class_output_network_description),
    'loss/batchwise_subsample': loss_params['batchwise_subsample'],
    'loss/positive_overlap': loss_params['positive_overlap'],
    'loss/negative_overlap': loss_params['negative_overlap'],
    'loss/max_weight': loss_params['max_weight'],
    'loss/sigma': loss_params['sigma'],

    'anchor_network/allowed_border': anchor_network_params['allowed_border'],
    'anchor_network/store_anchors': anchor_network_params['store_anchors'],
    'anchor_network/allowed_border': anchor_network_params['allowed_border'],

    'model/allowed_border': anchor_network_params['allowed_border'],
    'model/store_anchors': anchor_network_params['store_anchors'],
    'model/allowed_border': anchor_network_params['allowed_border'],

    'roi_selector/num_keep_pre_nms': roi_selector_params['num_keep_pre_nms'],
    'roi_selector/num_keep_post_nms': roi_selector_params['num_keep_post_nms'],
    'roi_selector/thresh_nms': roi_selector_params['thresh_nms'],
    'roi_selector/val_num_keep_pre_nms': roi_selector_params['val_num_keep_pre_nms'],
    'roi_selector/val_num_keep_post_nms': roi_selector_params['val_num_keep_post_nms'],
    'roi_selector/val_thresh_nms': roi_selector_params['val_thresh_nms'],

    'model/num_keep_pre_nms': roi_selector_params['num_keep_pre_nms'],
    'model/num_keep_post_nms': roi_selector_params['num_keep_post_nms'],
    'model/thresh_nms': roi_selector_params['thresh_nms'],
    'model/val_num_keep_pre_nms': roi_selector_params['val_num_keep_pre_nms'],
    'model/val_num_keep_post_nms': roi_selector_params['val_num_keep_post_nms'],
    'model/val_thresh_nms': roi_selector_params['val_thresh_nms'],
    'model/trainable_parameter_count': trainable_parameter_count,
}


wrong_values = [
    (key, type(value), value)
    for key, value
    in flat_config.items()
    if not isinstance(value, (int, float, str, bool, torch.Tensor))]

assert not wrong_values, wrong_values

# %%
print(model_params)
print(model)
print('num params:', trainable_parameter_count)
print(model.feature_extractor.class_stride)


# %%
if loss_load_state_file:
    state_dict = torch.load(
        state_file_dir + loss_load_state_file, map_location=torch.device('cpu'))

    if ignore_load:
        loss_state = loss.state_dict()

        pretrained_state = {
            k:v for k,v in state_dict.items() if k in loss_state}
        
        not_in_state = [
            key for key in loss_state.keys() if key not in pretrained_state]

        not_used = [
            key for key in pretrained_state.keys() if key not in loss_state]

        if not_in_state:
            print('Following parts ar not in state:')
            print(not_in_state)

        if not_used:
            print('Following parts ar not used:')
            print(not_in_state)
        loss_state.update(pretrained_state)
    else:
        loss_state = state_dict

    loss.load_state_dict(loss_state)

if load_state_file:
    state_dict = torch.load(
        state_file_dir + load_state_file, map_location=torch.device('cpu'))

    if ignore_load:
        model_state = model.state_dict()

        if load_shape_filter:
            pretrained_state = {
                k:v for k,v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape}
        else:
            pretrained_state = {
                k:v for k,v in state_dict.items() if k in model_state}
        # assert all(
        #     v.size() == model_state[k].size() for k,v in state_dict.items())

        not_in_state = [
            key for key in model_state.keys() if key not in pretrained_state]

        not_used = [
            key for key in pretrained_state.keys() if key not in model_state]

        if not_in_state:
            print('Following parts ar not in state:')
            print(not_in_state)

        if not_used:
            print('Following parts ar not used:')
            print(not_in_state)
        model_state.update(pretrained_state)
    else:
        model_state = state_dict

    model.load_state_dict(model_state)

# %%

if unet_tweak:
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    model.calc_segmentation = False
    model.feature_extractor.calc_unet = False

if mask_only:
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.bbox_network.parameters():
        param.requires_grad = False
    for param in model.class_network.parameters():
        param.requires_grad = False
    parameters = model.mask_network.parameters()
else:
    parameters = model.parameters()


dense_params = list(model.feature_extractor.anchor_networks.parameters())

if model.bbox_network is not None:
    dense_params += list(model.bbox_network.parameters())

sparse_params = [
    p for p in model.parameters()
    if p.requires_grad and not any(p is d for d in dense_params)]

learnable_parameter = [
    {'params': sparse_params},
    {'params': dense_params},
    {'params': loss.parameters()}]

if sgd:
    optimizer = optim.SGD(learnable_parameter, **optim_params)
else:
    optimizer = optim.Adam(learnable_parameter, **optim_params)

if scheduler_params is not None:
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    if continue_training is not None:
        for _ in range(start_epoch):
            scheduler.step()
else:
    scheduler = None

error_threshold = 0.1

# %%
if overfit:
    log_storage = log_storage[:-1] + 'Overfit/'
    model_storage = model_storage[:-1] + 'Overfit/'

# %%
