
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import numpy as np
from ndsis.modules.model import (
    ClassLossSelector, DenseMaskLossSelector, SparseMaskLossSelector,
    OverlapCalculator)


class Loss(nn.Module):
    def __init__(
            self,
            sigma=2.,
            class_target_overlap_threshold=0.15,
            batchwise_subsample=False,
            class_weights=None,
            mask_class_weights=None,
            class_class_weights=None,
            segmentation_weights=None,
            sparse=False,
            loss_rpn_class_weight=1,
            loss_rpn_bbox_weight=1,
            loss_class_weight=1,
            loss_mask_weight=1,
            loss_segmentation_weight=1,
            multitask_loss=False,
            dtype=torch.get_default_dtype(),
            *,
            calc_bbox,
            calc_class,
            calc_mask,
            calc_segmentation,
            mask_positive_threshold,
            class_positive_threshold,
            class_negative_threshold,
            class_negative_label,
            **bbox_target_selector_kwargs,):
        super(Loss, self).__init__()
        self.default_loss = nn.Parameter(
            torch.zeros([], dtype=dtype), requires_grad=False)

        self.loss_rpn_class_weight = nn.Parameter(
            -torch.log(torch.tensor(loss_rpn_class_weight, dtype=dtype)),
            requires_grad=multitask_loss)
        self.loss_rpn_bbox_weight = nn.Parameter(
            -torch.log(torch.tensor(loss_rpn_bbox_weight, dtype=dtype)),
            requires_grad=multitask_loss)
        self.loss_class_weight = nn.Parameter(
            -torch.log(torch.tensor(loss_class_weight, dtype=dtype)),
            requires_grad=multitask_loss)
        self.loss_mask_weight = nn.Parameter(
            -torch.log(torch.tensor(loss_mask_weight, dtype=dtype)),
            requires_grad=multitask_loss)
        self.loss_segmentation_weight = nn.Parameter(
            -torch.log(torch.tensor(loss_segmentation_weight, dtype=dtype)),
            requires_grad=multitask_loss)

        bbox_target_selector = (
            BatchwiseBboxTargetSelector(**bbox_target_selector_kwargs)
            if batchwise_subsample else
            SamplewiseBboxTargetSelector(**bbox_target_selector_kwargs))

        if calc_bbox:
            self.rpn_loss = RpnLoss(bbox_target_selector, sigma)
        else:
            self.rpn_loss = None

        if calc_class or calc_mask:
            self.overlap_calculator = OverlapCalculator()
        else:
            self.overlap_calculator = None

        if calc_class:
            self.class_loss_selector = ClassLossSelector(
                class_positive_threshold, class_negative_threshold,
                class_negative_label)
            self.class_loss = ClassLoss(class_weights=class_class_weights)
        else:
            self.class_loss_selector = None
            self.class_loss = None

        if calc_mask:
            self.mask_loss_selector = (
                SparseMaskLossSelector if sparse else DenseMaskLossSelector)(
                    mask_positive_threshold)
            self.mask_loss = MaskLoss(
                class_weights=mask_class_weights, dtype=dtype)
        else:
            self.mask_loss_selector = None
            self.mask_loss = None

        if calc_segmentation:
            self.segmentation_loss = nn.CrossEntropyLoss(
                weight=segmentation_weights, ignore_index=-100,
                reduction='mean')
        else:
            self.segmentation_loss = None

    def forward(self, batch, outputs):
        loss = self.default_loss.clone()
        multitask_extra_loss = self.default_loss.clone()
        sublosses = dict()

        if 'rpn_score' in outputs and 'rpn_bbox' in outputs and self.rpn_loss is not None:
            rpn_score_loss, rpn_bbox_loss = self.rpn_loss(
                batch['gt_bbox'], outputs['rpn_target_calculator'],
                outputs['rpn_score'], outputs['rpn_bbox'])

            loss += rpn_score_loss * torch.exp(-self.loss_rpn_bbox_weight)
            loss += rpn_bbox_loss * torch.exp(-self.loss_rpn_class_weight)
            multitask_extra_loss += self.loss_rpn_bbox_weight
            multitask_extra_loss += self.loss_rpn_class_weight

            sublosses = dict(
                sublosses,
                rpn_score=rpn_score_loss.item(),
                rpn_bbox=rpn_bbox_loss.item())

        if (
                ('mpn_class' in outputs and self.class_loss is not None) or
                ('mpn_mask' in outputs and self.mask_loss is not None)):
            if not outputs['is_training']:
                pred_gt_max_argmax_tuple_list = self.overlap_calculator(
                    outputs['roi_bbox'], batch['gt_bbox'])
            else:
                pred_gt_max_argmax_tuple_list = None

        if 'mpn_class' in outputs and self.class_loss is not None:
            selected_class_pred, selected_class_gts = self.class_loss_selector(
                outputs['mpn_class'],
                outputs['mpn_class_coord_selection'],
                outputs['mpn_class_box_selection'],
                pred_gt_max_argmax_tuple_list,
                batch['gt_label'])

            if len(selected_class_pred):
                class_loss = self.class_loss(
                    selected_class_pred, selected_class_gts)

                loss += class_loss * torch.exp(-self.loss_class_weight)
                multitask_extra_loss += self.loss_class_weight

                sublosses = {
                    **sublosses,
                    'class': class_loss.item()}

        if 'mpn_mask' in outputs and self.mask_loss is not None:
            (
                selected_mask_pred, selected_mask_gts, selected_mask_labels
            ) = self.mask_loss_selector(
                outputs['mpn_mask'],
                outputs['mpn_mask_coord_selection'],
                outputs['mpn_mask_box_selection'],
                pred_gt_max_argmax_tuple_list,
                batch['gt_label'],
                batch['gt_mask'])

            if len(selected_mask_pred):
                mask_loss = self.mask_loss(
                    selected_mask_pred, selected_mask_gts, selected_mask_labels)

                loss += mask_loss * torch.exp(-self.loss_mask_weight)
                multitask_extra_loss += self.loss_mask_weight

                sublosses = {
                    **sublosses,
                    'mask': mask_loss.item()}

        if 'spn_segmentation' in outputs and self.segmentation_loss is not None:

            segmentation_loss = self.segmentation_loss(
                outputs['spn_segmentation'], batch['gt_segmentation'])

            loss += segmentation_loss * torch.exp(
                -self.loss_segmentation_weight)
            multitask_extra_loss += self.loss_segmentation_weight

            sublosses = {
                **sublosses,
                'segmentation': segmentation_loss.item()}

        combined_loss = loss + multitask_extra_loss

        sublosses = {
            '_total_': loss.item(),
            '_total_multitask_': combined_loss.item(),
            **sublosses}

        return combined_loss, sublosses

    def get_weights(self):
        weights_dict = dict()
        if self.rpn_loss is not None:
            weights_dict['rpn_score'] = torch.exp(
                -self.loss_rpn_class_weight).item()
            weights_dict['rpn_bbox'] = torch.exp(
                -self.loss_rpn_bbox_weight).item()
        if self.class_loss is not None:
            weights_dict['class'] = torch.exp(
                -self.loss_class_weight).item()
        if self.mask_loss is not None:
            weights_dict['mask'] = torch.exp(
                -self.loss_mask_weight).item()
        if self.segmentation_loss is not None:
            weights_dict['segment'] = torch.exp(
                -self.loss_segmentation_weight).item()
        return weights_dict


class RpnLoss(nn.Module):
    def __init__(self, bbox_target_selector, sigma):
        super().__init__()
        self.bbox_target_selector = bbox_target_selector
        self.sigma = sigma

    def forward(self, gt_bbox, rpn_target_calculator, rpn_score, rpn_bbox):
        rpn_target_overlaps, rpn_target_association, rpn_bbox_target = \
            rpn_target_calculator(gt_bbox)

        rpn_score_target, rpn_score_weight, rpn_bbox_weights = \
            self.bbox_target_selector(rpn_target_overlaps)

        rpn_score_loss = F.binary_cross_entropy_with_logits(
            rpn_score,
            rpn_score_target,
            rpn_score_weight,
            reduction='sum'
        )

        rpn_bbox_loss = self.smooth_l1_loss(
            rpn_bbox,
            rpn_bbox_target,
            rpn_bbox_weights,
            )

        return rpn_score_loss, rpn_bbox_loss

    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_weights):
        sigma_2 = self.sigma ** 2
        box_diff = bbox_pred - bbox_targets

        abs_box_diff = torch.abs(box_diff)
        smoothL1_sign = ((abs_box_diff < 1. / sigma_2)
                         .detach().type(abs_box_diff.dtype))
        in_loss_box = (
            box_diff * box_diff * (sigma_2 / 2.) * smoothL1_sign
            + (abs_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign))
        loss_box = bbox_weights[..., None, None] * in_loss_box

        return loss_box.sum()


class ClassLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=-100, reduction='mean')

    def forward(self, class_output, class_target):
        class_target_cat = torch.cat(class_target)
        class_output_cat = torch.cat(class_output)
        if len(class_output_cat):
            loss = self.loss(class_output_cat, class_target_cat)
        else:
            loss = class_output_cat.new_zeros(())
        return loss


class MaskLoss(nn.Module):
    def __init__(self, class_weights=None, dtype=torch.get_default_dtype()):
        super().__init__()

        self.class_weights = (
            None
            if class_weights is None else
            nn.Parameter(class_weights, requires_grad=False))
        self.default_loss = nn.Parameter(
            torch.zeros([], dtype=dtype), requires_grad=False)
        self.mask_loss_reduction = 'mean'
        self.weight_range = 16

    def _single_sample_loss(self, output_mask, target_mask):
        target_mask_dtype = target_mask.type(output_mask.dtype)

        return F.binary_cross_entropy_with_logits(
            output_mask, target_mask_dtype,
            reduction=self.mask_loss_reduction)

    def forward(self, masks_output, mask_target, class_target):
        loss = self.default_loss
        class_target_cat = torch.cat(class_target)

        maskwise_loss_list = [
            self._single_sample_loss(single_output_mask, single_target_mask)
            for sample_mask_output, sample_mask_target
            in zip(masks_output, mask_target)
            for single_output_mask, single_target_mask
            in zip(sample_mask_output, sample_mask_target)]

        if maskwise_loss_list:
            maskwise_loss = torch.stack(maskwise_loss_list)
            valid = ~torch.isnan(maskwise_loss)
            valid_maskwise_loss = maskwise_loss[valid]
            valid_class_target_cat = class_target_cat[valid]

            if len(valid_maskwise_loss):
                if self.class_weights is not None:
                    mask_weights = (
                        self.class_weights[valid_class_target_cat])
                    loss = (
                        (valid_maskwise_loss * mask_weights).sum() /
                        mask_weights.sum())
                else:
                    loss = valid_maskwise_loss.mean()

        return loss


class SamplewiseBboxTargetSelector(nn.Module):
    def __init__(
            self,
            positive_overlap=0.35,
            negative_overlap=0.15,
            fraction_pos_per_neg=0.5,
            num_bbox_target_select=64):
        super(SamplewiseBboxTargetSelector, self).__init__()
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.fraction_pos_per_neg = fraction_pos_per_neg
        self.num_bbox_target_select = num_bbox_target_select

    def forward(self, overlaps):
        batch_pos_overlaps = overlaps >= self.positive_overlap
        batch_neg_overlaps = overlaps < self.negative_overlap

        batch_labels = batch_pos_overlaps.type(overlaps.dtype)
        batch_label_valid = torch.zeros_like(overlaps)

        for pos_overlaps, neg_overlaps, label_valid in zip(
                batch_pos_overlaps, batch_neg_overlaps, batch_label_valid):
            pos_indices, = torch.where(pos_overlaps)
            neg_indices, = torch.where(neg_overlaps)
            pos_count = len(pos_indices)
            neg_count = len(neg_indices)

            if self.num_bbox_target_select:
                pos_num_goal = int(self.fraction_pos_per_neg *
                                   self.num_bbox_target_select)
                pos_num_sample = min(pos_num_goal, pos_count)
                neg_num_goal = self.num_bbox_target_select - pos_num_sample
                neg_num_sample = min(neg_num_goal, neg_count)
            elif not self.fraction_pos_per_neg:
                pos_num_sample = pos_count
                neg_num_sample = neg_count
            elif self.fraction_pos_per_neg == 0.5:
                sample_amount = min(pos_count, neg_count)
                pos_num_sample = sample_amount
                neg_num_sample = sample_amount
            else:
                double_fraction = 2 * self.fraction_pos_per_neg
                pos_num_sample = int(
                    min(pos_count, neg_count / double_fraction))
                neg_num_sample = int(
                    min(pos_count * double_fraction, neg_count))

            if pos_num_sample:
                pos_index_indices = np.random.choice(
                    len(pos_indices), size=pos_num_sample, replace=False)
                label_valid[pos_indices[pos_index_indices]] = 1

            if neg_num_sample:
                neg_index_indices = np.random.choice(
                    len(neg_indices), size=neg_num_sample, replace=False)
                label_valid[neg_indices[neg_index_indices]] = 1

        batch_score_weight = (
            batch_label_valid /
            batch_label_valid.sum(-1, keepdim=True).clamp(min=1) /
            len(batch_label_valid))
        batch_bbox_weights = (
            batch_labels /
            batch_labels.sum(-1, keepdim=True).clamp(min=1) /
            len(batch_labels))

        return batch_labels, batch_score_weight, batch_bbox_weights


class BatchwiseBboxTargetSelector(nn.Module):
    def __init__(
            self,
            positive_overlap=0.35,
            negative_overlap=0.15,
            max_weight=(1/8)):
        super(BatchwiseBboxTargetSelector, self).__init__()
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.min_inverse_weight = 1 / max_weight

    def forward(self, overlaps):
        pos_overlaps = overlaps >= self.positive_overlap
        neg_overlaps = overlaps < self.negative_overlap

        pos_indices = torch.nonzero(pos_overlaps)
        neg_indices = torch.nonzero(neg_overlaps)
        pos_count = len(pos_indices)
        neg_count = len(neg_indices)

        max_count = max(pos_count, neg_count)
        min_count = min(pos_count, neg_count)

        subsample = np.random.choice(max_count, size=min_count, replace=False)

        if pos_count > neg_count:
            batch_score_weight = neg_overlaps.type(overlaps.dtype)
            sampled_indices = pos_indices[subsample]
        else:
            batch_score_weight = pos_overlaps.type(overlaps.dtype)
            sampled_indices = neg_indices[subsample]

        batch_score_weight[sampled_indices.unbind(-1)] = 1
        batch_score_weight /= max(1, (2 * min_count))

        batch_labels = pos_overlaps.type(overlaps.dtype)

        batch_bbox_weights = (
            batch_labels /
            batch_labels.sum().clamp(min=self.min_inverse_weight))

        return batch_labels, batch_score_weight, batch_bbox_weights


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        if reduction == 'mean':
            self.reduction_method = torch.mean
        elif reduction == 'sum':
            self.reduction_method = torch.sum
        else:
            self.reduction_method = lambda x: x

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='None')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return self.reduction_method(focal_loss)