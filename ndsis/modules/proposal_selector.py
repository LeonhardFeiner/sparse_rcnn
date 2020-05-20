import torch
import torch.nn as nn
from ndsis.utils.bbox import non_maximum_supression
from ndsis.modules.custom_container import ConditionalStage

def get_roi_selector(
        num_keep_pre_nms=1000,
        num_keep_post_nms=500,
        thresh_nms=0.5,
        val_num_keep_pre_nms=None,
        val_num_keep_post_nms=None,
        val_thresh_nms=None,):
    roi_selector = RoiSelector(
        num_keep_pre_nms, num_keep_post_nms, thresh_nms)

    if val_num_keep_pre_nms:
        val_roi_selector = RoiSelector(
            val_num_keep_pre_nms, val_num_keep_post_nms, val_thresh_nms)
        roi_selector = ConditionalStage(roi_selector, val_roi_selector)
    
    return roi_selector


class RoiSelector(nn.Module):
    def __init__(self, *args, detach=True, **kwargs):
        super().__init__()
        self.proposal_selector = ProposalSelector(*args, **kwargs)
        self.detach = detach

    def forward(self, rpn_bbox, rpn_score, anchor_description):
        if self.detach:
            rpn_bbox = rpn_bbox.detach()
            rpn_score = rpn_score.detach()
        # [batch x inside_anchors x 2 x spatial_dims]
        roi_bbox_raw = anchor_description(rpn_bbox)
        # [batch x inside_anchors]
        roi_score_raw = torch.sigmoid(rpn_score)

        # num_rois are the amount of regions of interest which were
        #  not removed by non maximum supression. Varies for each sample.
        # batch x [num_rois x 2 x spatial_dims]
        # roi_bbox
        # batch x [num_rois]
        # roi_score
        # batch x [num_rois]
        # roi_level
        roi_score, roi_bbox, roi_index = self.proposal_selector(
            roi_score_raw, roi_bbox_raw)

        return roi_score, roi_bbox, roi_index


class ProposalSelector(nn.Module):
    def __init__(self, num_keep_pre_nms, num_keep_post_nms, thresh_nms):
        super(ProposalSelector, self).__init__()
        self.num_keep_pre_nms = num_keep_pre_nms
        self.num_keep_post_nms = num_keep_post_nms
        self.thresh_nms = thresh_nms

    def forward(self, rpn_score, rpn_bbox):
        if self.num_keep_pre_nms > 0:
            rpn_score, indices = torch.topk(
                rpn_score, self.num_keep_pre_nms, dim=1, sorted=True)
        else:
            rpn_score, indices = torch.sort(rpn_score, dim=1, descending=True)

        indices = indices.cpu()
        batch_index = torch.arange(len(rpn_bbox)).unsqueeze(1)
        rpn_bbox = rpn_bbox[batch_index, indices]

        # non maximum suppression
        nms_indicator = non_maximum_supression(rpn_bbox, self.thresh_nms)
        # selection of the boxes which are the maximum at its position
        rpn_score = [
            sample_scores[sample_keep][:self.num_keep_post_nms]
            for sample_scores, sample_keep
            in zip(rpn_score, nms_indicator)]

        rpn_bbox = [
            sample_rpn_bbox[sample_keep][:self.num_keep_post_nms]
            for sample_rpn_bbox, sample_keep
            in zip(rpn_bbox, nms_indicator)]

        rpn_index = [
            sample_indices[sample_keep][:self.num_keep_post_nms]
            for sample_indices, sample_keep
            in zip(indices, nms_indicator)]

        return rpn_score, rpn_bbox, rpn_index
