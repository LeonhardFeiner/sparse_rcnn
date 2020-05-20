from typing import Callable, Tuple, List
from collections.abc import Iterable
from itertools import chain
import torch
from torch.nn.functional import pad
import numpy as np
from ndsis.utils.basic_functions import cummax_reversed
from ndsis.utils.bbox import bbox_overlap_prediction as bbox_iou
from ndsis.utils.mask import mask_iou_matrix, mask_confusion_pair, select_mask, mask_iou_matrix_split_combine


class OverlapCalculator():
    """Overlap_Calculator calculates the overlaps between ground truth
        and prediction.

    Args:
        iou_calculator (Callable[[torch.tensor, torch.tensor], torch.tensor]):
            the function which calculates the overlap
        score_threshold (int, optional): Defaults to None
            predictions below this threshold are removed before calculation.
            no predictions are removed if score_threshold is None.
        sort (bool, optional): Defaults to False.
            indicates whether predictions have to be sorted or are presorted.
    """
    def __init__(
            self,
            iou_calculator: Callable[
                [torch.tensor, torch.tensor], torch.tensor],
            score_threshold: int = None,
            sort: bool = False):
        super().__init__()
        self.iou_calculator = iou_calculator
        self.score_threshold = score_threshold
        self.sort = sort

    def __call__(
        self,
        score: torch.tensor,
        pred: torch.tensor,
        gt: torch.tensor,
        pred_class: torch.tensor = False) -> Tuple[
            torch.tensor, torch.tensor, torch.tensor]:
        """calculate the sorted overlap matrix between prediction and ground
            truth.

        remaining_shape is defined by the iou_calculator function

        Args:
            score (torch.tensor): (N_{predictions})
                the score of the predictions
            pred (torch.tensor): (N_{predictions}, *remaining_shape)
                the prediction itself. Type of prediction depends on
                iou_calculator.
            gt (torch.tensor): (N_{ground_truth}, *remaining_shape)
                the ground truth data to compare the predictions with
            pred_class (torch.LongTensor, optional): (N_{predictions})
                defaults to None. predictions of the classes if there are any.

        Returns:
            score (torch.tensor): (N_{filtered_predictions})
                the filtered and sorted scores
            iou (torch.tensor): (N_{filtered_predictions}, N_{ground_truth})
                the iou matrix which compares predictions to its ground truth.
            pred_class (Union[torch.LongTensor, None):
                (N_{filtered_predictions}) returns the predicted class of the
                filtered and sorted predictions if given
        """
        score = score.detach()
        pred = pred.detach()
        if self.score_threshold is not None:
            valid_indicator = score >= self.score_threshold
            score = score[valid_indicator]
            pred = pred[valid_indicator]
            if pred_class is not None:
                pred_class = pred_class[valid_indicator]
        if self.sort:
            score, indices = score.sort(descending=True)
            pred = pred[indices]
            if pred_class is not None:
                pred_class = pred_class[indices]

        iou = self.iou_calculator(pred, gt)
        return score, iou, pred_class


class MaskOverlapCalculator(OverlapCalculator):
    """Mask_Overlap_Calculator calculates the overlaps between ground truth
        and prediction masks.

    Args:
        mask_threshold
        score_threshold (int, optional): Defaults to None
            predictions below this threshold are removed before calculation.
            no predictions are removed if score_threshold is None.
        sort (bool, optional): Defaults to False.
            indicates whether predictions have to be sorted or are presorted.
        filter_masks (bool, optional): Defaults to False.
            indicates whether masks which don't contain any predicted item
            should be removed before evaluation.
    """
    def __init__(
            self, mask_threshold=0.5, score_threshold=None, sort=False,
            filter_masks=False):
        super().__init__(
            mask_iou_matrix_split_combine, score_threshold=score_threshold,
#            mask_iou_matrix, score_threshold=score_threshold,
            sort=sort)
        self.mask_threshold = mask_threshold
        self.filter_masks = filter_masks

    def __call__(self, score, pred, gt, pred_class):
        """calculate the sorted overlap matrix between prediction and ground
            truth.

        Args:
            score (torch.tensor): (N_{predictions})
                the score of the predictions
            pred (torch.tensor): (N_{predictions}, *)
                the predicted masks.
            gt (torch.tensor): (N_{ground_truth}, *)
                the ground truth masks.
            pred_class (torch.LongTensor, optional): (N_{predictions})
                defaults to None. predictions of the classes if there are any.

        Returns:
            score (torch.tensor): (N_{filtered_predictions})
                the filtered and sorted scores
            iou (torch.tensor): (N_{filtered_predictions}, N_{ground_truth})
                the iou matrix which compares predictions masks to its ground
                truth masks.
            pred_class (Union[torch.LongTensor, None):
                (N_{filtered_predictions}) returns the predicted class of the
                filtered and sorted predictions if given
        """
        bool_pred = pred > self.mask_threshold

        if self.filter_masks:
            valid_masks = bool_pred
            for _ in range(bool_pred.ndim - 1):
                valid_masks = valid_masks.any(axis=1)

            valid_bool_pred = bool_pred[valid_masks]
            valid_scores = score[valid_masks]
            valid_pred_class = pred_class[valid_masks]
            return super().__call__(
                valid_scores, valid_bool_pred, gt, valid_pred_class)
        else:
            return super().__call__(score, bool_pred, gt, pred_class)


class BboxOverlapCalculator(OverlapCalculator):
    """Mask_Overlap_Calculator calculates the overlaps between ground truth
        and prediction masks.

    Args:
        score_threshold (int, optional): Defaults to None
            predictions below this threshold are removed before calculation.
            no predictions are removed if score_threshold is None.
        sort (bool, optional): Defaults to False.
            indicates whether predictions have to be sorted or are presorted.
    """
    def __init__(self, score_threshold=None, sort=False):
        super().__init__(
            bbox_iou, score_threshold=score_threshold, sort=sort)

    def __call__(self, score, pred, gt, pred_class):
        """calculate the sorted overlap matrix between prediction and ground
            truth boxes.

        Args:
            score (torch.tensor): (N_{predictions})
                the score of the predictions
            pred (torch.tensor): (N_{predictions}, 2, N_{dims})
                the predicted boxes
            gt (torch.tensor): (N_{ground_truth}, 2, N_{dims})
                the ground truth boxes
            pred_class (torch.LongTensor, optional): (N_{predictions})
                defaults to None. predictions of the classes if there are any.

        Returns:
            score (torch.tensor): (N_{filtered_predictions})
                the filtered and sorted scores
            iou (torch.tensor): (N_{filtered_predictions}, N_{ground_truth})
                the iou matrix which compares predictions masks to its ground
                truth masks.
            pred_class (Union[torch.LongTensor, None):
                (N_{filtered_predictions}) returns the predicted class of the
                filtered and sorted predictions if given
        """
        return super().__call__(score, pred, gt, pred_class)


class OverlapAccumulator():
    """OverlapAccumulator stores the overlap matrices between ground truth and
    prediction of a variable amount of batches. It further can be used to store
    ground truth semantic classes and predicted semantic class. It is required
    to calculate a pr curve without classes or a pr curve collection of a set
    of classes. 

    Args:
        overlap_calculator (
            Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor],
            torch.tensor]):
            the function which filters the predictions and calculates the
            overlap matrices
        device (Union[torch.device, string], optional): Defaults to 'cpu'
            the device, where the iou matrices should be stored after
            calculation.
    """
    def __init__(self, overlap_calculator, device='cpu'):
        super().__init__()
        self.overlap_calculator = overlap_calculator
        self.device = torch.device(device)
        self.score_list = list()
        self.iou_list = list()
        self.pred_class_list = list()
        self.gt_class_list = list()

    def add_sample(self, score, pred, gt, pred_class=None, gt_class=None):
        """calculate the overlap matrix between prediction and ground truth
        boxes of a single scene and store them into the accumulator.

        Args:
            score (torch.tensor): (N_{predictions})
                the score of the predictions
            pred (torch.tensor): (N_{predictions}, *)
                the predicted boxes, masks, etc.
            gt (torch.tensor): (N_{ground_truth}, *)
                the ground truth boxes
            pred_class (torch.LongTensor, optional): (N_{predictions})
                defaults to None. predictions of the classes if there are any.
            gt_class (torch.LongTensor, optional): (N_{predictions})
                defaults to None. labels of the classes if there are any.
        """
        score, iou, pred_class = self.overlap_calculator(
            score, pred, gt, pred_class)
        self.score_list.append(score.to(self.device))
        self.iou_list.append(iou.to(self.device))
        if pred_class is None and gt_class is None:
            self.pred_class_list.append(None)
            self.gt_class_list.append(None)
        else:
            self.pred_class_list.append(pred_class.to(self.device))
            self.gt_class_list.append(gt_class.to(self.device))

    def add_batch(self, *args):
        """calculate the overlap matrix between prediction and ground truth
        boxes of a single scene and store them into the accumulator.

        Args:
            score (List[torch.tensor]): N_scenes x (N_{predictions})
                a list containing the score of the predictions of single scenes
            pred (List[torch.tensor]):  N_scenes x (N_{predictions}, *)
                a list containing the predictions of single scenes
            gt (List[torch.tensor]):  N_scenes x (N_{ground_truth}, *)
                a list containing the the ground truth pf single scenes
            pred_class (List[torch.LongTensor], optional): 
                N_scenes x (N_{predictions})
                defaults to None. a list containing predictions of the classes
                of single scenes if there are any.
            gt_class (List[torch.LongTensor], optional): 
                N_scenes x (N_{predictions})
                defaults to None. a list containing labels of the classes of
                single scenes if there are any.
        """
        for sample_args in zip(*args):
            self.add_sample(*sample_args)

    def get_pr_curve(self, overlap_threshold, device=None):
        """calculate a pr curve without taking classes into account.

        Args:
            overlap_threshold (float): the minimal overlap which is encauntert
                as match between prediction and ground truth
            device (Union[torch.device, string], optional): Defaults to None.
                the device, where the pr curves should be calculated

        Returns:
            pr_curve (PrecisionRecallCurve): The precision recall curve object
                calculated with the given overlap threshold.
        """
        return PrecisionRecallCurve.create_by_sorted_tensor_lists(
             self.score_list, self.iou_list, overlap_threshold, device)

    def get_classwise_accumulator(self, classes, device=None):
        """separates the overlaps by predicted and ground truth class.

        The resulting ClasswiseOverlapAccumulator contains one overlap 

        Args:
            classes (torch.LongTensor): the classes which should be evaluated
            device (Union[torch.device, string], optional): Defaults to None.
                the device, where the ClasswiseOVerlapAccumulator should be
                stored

        Returns:
            classwise_overlap_accumulator (ClasswiseOverlapAccumulator):
                The overlap accumulator which separates by classes
        """
        return ClasswiseOverlapAccumulator(
            self.score_list, self.iou_list, self.pred_class_list,
            self.gt_class_list, classes, device)

    def has_classes(self):
        """Checks wether predictions contain classes.

        If the accumulator contains classes, a classwise accumulator can be
        calculated.
        
        Returns:
            contains_classes (bool): indicator which describes if the
            prediction contain classes
        """
        return all(pred is not None for pred in self.pred_class_list)

    def get_counts(self):
        return np.array(
            [iou.shape for iou in self.iou_list])


class ClasswiseOverlapAccumulator():
    """ClasswiseOverlapAccumulator stores the overlap matrices between ground
    truth and prediction separated by classes of a variable amount of scenes.

    It is required to calculate a pr curve collection of a set of classes.

    Args:
        score_list (List[torch.tensor]): N_classes x (N_{filtered_predictions})
            the list of filtered and sorted scores
        iou (List[torch.tensor]):
            N_classes x (N_{filtered_predictions}, N_{ground_truth})
            the list of iou matrices which compares predictions masks to its
            ground truth masks.
        device (Union[torch.device, string], optional): Defaults to 
            the device, where the iou matrices should be stored after
            separation.
    """
    @staticmethod
    def select_classes(score, iou, pred_class, gt_class, classes, device=None):
        """separates the overlap matrices by classes.

        It is required to calculate a pr curve collection of a set of classes. 

        Args:
            score ([List[torch.tensor]):
                N_classes x N_scenes x (N_{filtered_predictions})
                the filtered and sorted scores
            iou_list (List[List[torch.tensor]]):
                N_classes x N_scenes x (N_{filtered_predictions}, N_{ground_truth})
                the iou matrix which compares predictions masks to its ground
                truth masks.
            device (Union[torch.device, string], optional): Defaults to 'cpu'
                the device, where the iou matrices should be stored after
                separation.
        """
        score = score.to(device)
        iou = iou.to(device)
        associated_pred = classes == pred_class.to(device)
        associated_gt = classes == gt_class.to(device)

        selected_score = [
            score[single_class_associated_pred]
            for single_class_associated_pred
            in associated_pred]
        selected_iou = [
            iou[single_class_associated_pred][:, single_class_associated_gt]
            for single_class_associated_pred, single_class_associated_gt
            in zip(associated_pred, associated_gt)]

        return selected_score, selected_iou

    def __init__(
            self, score_list, iou_list, pred_class_list, gt_class_list,
            classes, device=None):
        classes = torch.as_tensor(
            classes, device=device, dtype=torch.long).unsqueeze(1)

        class_score_list_list_unzipped, class_iou_list_list_unzipped = zip(*(
            ClasswiseOverlapAccumulator.select_classes(
                score, iou, pred_class, gt_class, classes, device)
            for score, iou, pred_class, gt_class
            in zip(score_list, iou_list, pred_class_list, gt_class_list)))

        self.sorted_score_list_list = list(zip(
            *class_score_list_list_unzipped))
        self.sorted_iou_list_list = list(zip(*class_iou_list_list_unzipped))

    def get_pr_collection(self, overlap_threshold, device=None):
        """calculate a collection of pr curves separated by classes.

        Args:
            overlap_threshold (float): the minimal overlap which is encauntert
                as match between prediction and ground truth
            device (Union[torch.device, string], optional): Defaults to None.
                the device, where the pr curves should be calculated

        Returns:
            pr_curve_class_collection (PrecisionRecallCurveClassCollection):
                The precision recall curve collection object calculated with
                the given overlap threshold.
        """
        return (
            PrecisionRecallCurveClassCollection
            .create_by_sorted_tensor_list_lists(
                self.sorted_score_list_list,
                self.sorted_iou_list_list,
                overlap_threshold, device))

    def get_class_counts(self):
        """calculate a list of numbers of ground truth objects per class.

        Returns:
            gt_count_array_list (List[np.array]):
                The number of gt objects per class and per scene
        """
        return np.array([
            [iou.shape for iou in iou_list]
            for iou_list in self.sorted_iou_list_list])


class PrecisionRecallCurveClassCollection(list):
    """PrecisionRecallCurveClassCollection stores a list precision recall
    curves with one curve per class.

    It is required to calculate a pr curve collection of a set of classes.

    Args:
        pr_curve_list (Iterable[PrecisionRecallCurve]): N_classes
            a iterable containing one pr curve per class
    """
    @staticmethod
    def create_by_sorted_tensor_list_lists(
            sorted_score_list_list, sorted_iou_list_list, overlap_threshold,
            device=None):
        """factory method for PrecisionRecallCurveClassCollection

        It is required to calculate a pr curve collection of a set of classes.

        Args:
            sorted_score_list_list (List[List[torch.tensor]]):
                N_classes x N_scenes x (N_{filtered_predictions})
                the class-wise and scene-wise sorted scores
            sorted_iou_list_list (List[List[torch.tensor]]): 
                N_classes x N_scenes x (N_{filtered_predictions}, N_{ground_truth})
                the class-wise and scene-wise sorted iou matrices which
                contain the overlap of predictions with their ground truth.
            overlap_threshold (float): the minimal overlap which is encauntert
                as match between prediction and ground truth
            device (Union[torch.device, string], optional): Defaults to None.
                the device, where the pr curves should be calculated

        Returns:
            pr_curve_class_collection (PrecisionRecallCurveClassCollection):
                The precision recall curve collection object calculated with
                the given overlap threshold.
        """

        pr_curves = [
            PrecisionRecallCurve.create_by_sorted_tensor_lists(
                sorted_score_list, sorted_iou_list, overlap_threshold, device)
            for sorted_score_list, sorted_iou_list
            in zip(sorted_score_list_list, sorted_iou_list_list)]

        return PrecisionRecallCurveClassCollection(pr_curves)

    def get_ap_list_interpolated_all_points(self, no_gt_is_zero=False):
        """calculate the classwise average precision using the interpolated
        pr-curve.

        Args:
            no_gt_is_zero (bool): Defaults to False
                indicates whether classes without any ground truth objects
                should be return zero, otherwise they return nan

        Returns:
            ap_list (List[float]):
                the class-wise average precision.
        """
        return [
            pr_curve.get_ap_interpolated_all_points(
                no_gt_is_zero=no_gt_is_zero)
            for pr_curve in self]

    def get_ap_list_interpolated_sampled(
            self, num_samples=11, no_gt_is_zero=False):
        """calculate the classwise average precision using n samples of an
        interpolated pr-curve.

        Args:
            num_samples (int): Defaults to 11
                number of samples used to calculate the average precision.
            no_gt_is_zero (bool): Defaults to False
                indicates whether classes without any ground truth objects
                should be return zero, otherwise they return nan

        Returns:
            ap_list (List[float]):
                the class-wise average precision.
        """
        return [
            pr_curve.get_ap_interpolated_sampled(
                num_samples=num_samples, no_gt_is_zero=no_gt_is_zero)
            for pr_curve in self]

    def get_map_interpolated_all_points(self):
        """calculate the mean average precision using the interpolated
        pr-curves.

        Ignores classes without ground truth objects

        Returns:
            map (float):
                the mean average precision.
        """
        return np.nanmean(self.get_ap_list_interpolated_all_points())

    def get_map_interpolated_sampled(self, num_samples=11):
        """calculate the mean average precision using the sampled interpolated
        pr-curves.

        Ignores classes without ground truth objects

        Args:
            num_samples (int): Defaults to 11
                number of samples used to calculate the average precisions.

        Returns:
            map (float):
                the mean average precision.
        """
        return np.nanmean(self.get_ap_list_interpolated_sampled(
            num_samples=num_samples))


class PrecisionRecallCurve():
    """PrecisionRecallCurveClass stores a precision recall curve.

    Args:
        sorted_score (torch.tensor): (N_allscenes,)
            a sorted list of all prediction scores within the dataset
        sorted_tp_indicator (torch.BoolTensor): (N_allscenes,)
            a sorted list of all prediction indicator within the dataset
            every value indicates whether the prediction is a true positive or
            a false positive
        num_gt (int):
            the number of ground truth objects within the dataset
    """
    @staticmethod
    def create_by_sorted_tensors(
            sorted_score, sorted_iou, overlap_threshold, device=None):

        """factory method for PrecisionRecallCurve of a single scene

        Args:
            sorted_score (torch.tensor): (N_{filtered_predictions},)
                the sorted scores
            sorted_iou (torch.tensor):
                (N_{filtered_predictions}, N_{ground_truth})
                the sorted iou matrix which contains the overlap of
                predictions with their ground truth.
            overlap_threshold (float): the minimal overlap which is encauntert
                as match between prediction and ground truth
            device (Union[torch.device, string], optional): Defaults to None.
                the device, where the pr curves should be calculated

        Returns:
            pr_curve (PrecisionRecallCurve):
                The precision recall curve object calculated with the given
                overlap threshold.
        """

        tp_indicator = PrecisionRecallCurve.calc_tp_indicator(
            sorted_iou.to(device), overlap_threshold)
        num_gt = sorted_iou.shape[1]

        return PrecisionRecallCurve(
            sorted_score.to(device), tp_indicator, num_gt)

    @staticmethod
    def create_by_sorted_tensor_lists(
            sorted_score_list, sorted_iou_list, overlap_threshold,
            device=None):

        """factory method for PrecisionRecallCurve of a full dataset

        Args:
            sorted_score_list (List[torch.tensor]):
                N_scenes x (N_{filtered_predictions})
                the scene-wise sorted scores
            sorted_iou_list (List[torch.tensor]): 
                N_scenes x (N_{filtered_predictions}, N_{ground_truth})
                the scene-wise sorted iou matrices which contain the overlap of
                predictions with their ground truth.
            overlap_threshold (float): the minimal overlap which is encauntert
                as match between prediction and ground truth
            device (Union[torch.device, string], optional): Defaults to None.
                the device, where the pr curves should be calculated

        Returns:
            pr_curve (PrecisionRecallCurve):
                The precision recall curve object calculated with the given
                overlap threshold.
        """
        score = torch.cat(sorted_score_list).to(device)

        tp_indicator = torch.cat([
            PrecisionRecallCurve.calc_tp_indicator(
                sorted_sample_iou.to(device), overlap_threshold)
            for sorted_sample_iou in sorted_iou_list])

        num_gt = sum(sample_iou.shape[1] for sample_iou in sorted_iou_list)

        sorted_score, sorted_indices = score.sort(descending=True)
        sorted_tp_indicator = tp_indicator[sorted_indices]

        return PrecisionRecallCurve(sorted_score, sorted_tp_indicator, num_gt)

    @staticmethod
    def calc_tp_indicator(iou_sorted_by_score, overlap_threshold):
        """calculates a list of true or false positive indicator

        Args:
            iou_sorted_by_score (torch.tensor):
                (N_{predictions}, N_{ground_truth})
                the sorted iou matrix which contains the overlap of
                predictions with their ground truth.
            overlap_threshold (float): the minimal overlap which is encauntert
                as match between prediction and ground truth

        Returns:
            sorted_tp_indicator (torch.BoolTensor): (N_{predictions},)
                a sorted list of all prediction indicator within the dataset
                every value indicates whether the prediction is a true positive
                or a false positive
        """
        remaining_gt = list(range(iou_sorted_by_score.shape[1]))
        tp_indicator = torch.zeros(
            iou_sorted_by_score.shape[0], dtype=torch.bool)

        if remaining_gt:
            for single_iou, single_tp_indicator in zip(
                    iou_sorted_by_score, tp_indicator):
                max_iou, argmax_gt_iou = single_iou[remaining_gt].max(0)
                if max_iou >= overlap_threshold:
                    remaining_gt.pop(argmax_gt_iou.item())
                    single_tp_indicator[()] = True
                    if not remaining_gt:
                        break
        return tp_indicator.to(iou_sorted_by_score.device)

    @staticmethod
    def calc_pr_from_tp(
            sorted_tp_indicator, num_gt, dtype=torch.get_default_dtype()):
        """calculates the precision and the recall values of a pr curve

        Args:
            sorted_tp_indicator (torch.BoolTensor): (N_{predictions},)
                a sorted list of all prediction indicator within the dataset
                every value indicates whether the prediction is a true positive
                or a false positive
            num_gt (int):
                the number of ground truth objects which should be predicted
            dtype:
                the datatype used for calculation of the precision and recall
                values

        Returns:
            precision (torch.tensor): (N_{predictions},)
                the precision values of the precision recall curve
            recall (torch.tensor): (N_{predictions},)
                the recall values of the precision recall curve
        """
        tp_cumsum = sorted_tp_indicator.type(dtype).cumsum(0)
        precision = (
            tp_cumsum / torch.arange(
                1, 1 + len(tp_cumsum), dtype=dtype, device=tp_cumsum.device))
        recall = tp_cumsum / num_gt
        return precision, recall

    def __init__(self, sorted_score, sorted_tp_indicator, num_gt):
        precision, recall = self.calc_pr_from_tp(
            sorted_tp_indicator, num_gt, dtype=sorted_score.dtype)

        precision_interpolated = cummax_reversed(precision)

        self.score = sorted_score
        self.tp_indicator = sorted_tp_indicator
        self.num_gt = num_gt
        self.precision = precision
        self.recall = recall
        self.precision_interpolated = precision_interpolated

    def get_relevant_point_indicator(self):
        """calculates whether a point of the precision recall curve lies not
        in the center of a straight line and therefore at a corner of the
        curve. As only these points are required for visualization and for
        calculation, the others can be removed for compression.

        Returns:
            relevant_indicator (torch.BoolTensor): (N_{predictions},)
                a list of bools which indicates whether each point of the curve
                is required.
        """
        if len(self.tp_indicator):
            return (
                self.tp_indicator |
                pad(self.tp_indicator, (-1, 1), value=True))
        else:
            return self.tp_indicator

    def get_pr_raw(self, relevant_only=False):
        """returns the precision and the recall values of a pr curve

        Args:
            relevant_only (bool):
                an indicator whether the points of the pr curve should be
                filtered for compression purposes.

        Returns:
            precision (torch.tensor):
                (N_{predictions},) or (N_{relevant_predictions},)
                the precision values of the precision recall curve.
            recall (torch.tensor):
                (N_{predictions},) or (N_{relevant_predictions},)
                the recall values of the precision recall curve.
        """
        if relevant_only:
            relevant = self.get_relevant_point_indicator()
            return self.precision[relevant], self.recall[relevant]
        else:
            return self.precision, self.recall

    def get_pr_interpolated(self, relevant_only=False):
        """returns the interpolated precision and the recall values of a pr
        curve.

        Args:
            relevant_only (bool):
                an indicator whether the points of the pr curve should be
                filtered for compression purposes.

        Returns:
            precision (torch.tensor):
                (N_{predictions},) or (N_{relevant_predictions},)
                the interpolated precision values of the precision recall
                curve.
            recall (torch.tensor):
                (N_{predictions},) or (N_{relevant_predictions},)
                the recall values of the precision recall curve.
        """
        if relevant_only:
            relevant = self.get_relevant_point_indicator()
            return self.precision_interpolated[relevant], self.recall[relevant]
        else:
            return self.precision_interpolated, self.recall

    def get_sampled_interpolated_precision(self, num_samples=11):
        """returns the n samples of the interpolated precision of a pr curve.

        Args:
            num_samples (int): Defaults to 11
                number of samples used to take from the interpolated precision
                values.

        Returns:
            precision (torch.tensor): (num_samples,)
                the samples of the interpolated precision values of the
                precision recall curve.
        """
        samples = torch.linspace(
            0, 1, num_samples, dtype=self.recall.dtype,
            device=self.recall.device)
        is_greater_indicator = self.recall > samples.unsqueeze(1)
        first_indicator = is_greater_indicator.long().cumsum(1) == 1
        indices = torch.where(first_indicator)[1]

        selected_precision = self.precision_interpolated[indices]

        padded_precission = pad(
            selected_precision,
            (0, len(samples) - len(selected_precision)),
            value=0)

        return padded_precission

    def get_ap_interpolated_all_points(self, no_gt_is_zero=False):
        """calculate the average precision using the interpolated pr curve.

        Args:
            no_gt_is_zero (bool): Defaults to False
                indicates whether classes without any ground truth objects
                should be return zero, otherwise they return nan

        Returns:
            ap (float):
                the average precision.
        """
        if len(self.recall):
            recall_delta = self.recall - pad(self.recall, (1, -1))
            return (recall_delta * self.precision_interpolated).sum()
        elif self.num_gt or no_gt_is_zero:
            return torch.tensor(
                0, dtype=self.recall.dtype, device=self.recall.device)
        else:
            return torch.tensor(
                float('nan'),
                dtype=self.recall.dtype, device=self.recall.device)

    def get_ap_interpolated_sampled(self, num_samples=11, no_gt_is_zero=False):
        """calculate the average precision using n samples of an interpolated
        pr-curve.

        Args:
            num_samples (int): Defaults to 11
                number of samples used to calculate the average precision.
            no_gt_is_zero (bool): Defaults to False
                indicates whether classes without any ground truth objects
                should be return zero, otherwise they return nan

        Returns:
            ap (float):
                the average precision.
        """
        if self.num_gt or no_gt_is_zero:
            return self.get_sampled_interpolated_precision(num_samples).mean()
        else:
            return torch.tensor(
                float('nan'),
                dtype=self.recall.dtype, device=self.recall.device)

    def get_raw_curve(self, interpolated=False, relevant_only=True, flip=True):
        tp = self.tp_indicator.type(self.recall.dtype).cumsum(0)
        fp = (
            torch.arange(1, 1 + len(tp), dtype=tp.dtype, device=tp.device)
            - tp)
        tn = torch.zeros_like(tp)
        fn = self.num_gt - tp
        precision = (
            self.precision_interpolated if interpolated else self.precision)
        recall = self.recall

        if len(tp) and relevant_only:
            relevant = self.get_relevant_point_indicator()
            tp = tp[relevant]
            fp = fp[relevant]
            tn = tn[relevant]
            fn = fn[relevant]
            precision = precision[relevant]
            recall = recall[relevant]

        tp = tp.cpu()
        fp = fp.cpu()
        tn = tn.cpu()
        fn = fn.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        if flip:
            return (
                tp.flip((0,)), fp.flip((0,)), tn.flip((0,)), fn.flip((0,)),
                precision.flip((0,)), recall.flip((0,)), len(tp))
        else:
            return tp, fp, tn, fn, precision, recall, len(tp)


class EvaluationHelper():
    class DefaultNames():
        def __index__(self, idx):
            return idx

    @staticmethod
    def _get_name(name, metric_name, overlap_threshold, method):
        return (
            f'{name}_{metric_name}_{overlap_threshold}'
            f'{"" if method is None else "_{method}_points"}')

    @staticmethod
    def _get_ap(pr_curve, method):
        retval = (
            pr_curve.get_ap_interpolated_all_points()
            if method is None else
            pr_curve.get_ap_interpolated_sampled(method))

        return retval.cpu().item()

    @staticmethod
    def _get_ap_list(pr_collection, method):
        retval = (
            pr_collection.get_ap_list_interpolated_all_points()
            if method is None else
            pr_collection.get_ap_list_interpolated_sampled(method))

        return torch.stack(retval).cpu().numpy()

    def __init__(
            self, overlap_thresholds, overlap_class_indices,
            overlap_class_names=DefaultNames(), overlap_methods=[None, 11],
            confusion_class_names=DefaultNames(), device=None):
        super().__init__()

        self.single_output_overlap_tresholds = [
            single_overlap_treshold
            for single_overlap_treshold
            in overlap_thresholds
            if not isinstance(single_overlap_treshold, Tuple)]

        self.multiple_overlap_threshold_lists = [
            multiple_overlap_tresholds
            for multiple_overlap_tresholds
            in overlap_thresholds
            if isinstance(multiple_overlap_tresholds, Tuple)]

        multiple_overlap_threshold_list = (
            single_overlap_treshold
            for multiple_threshold_name, multiple_overlap_treshold_list
            in self.multiple_overlap_threshold_lists
            for single_overlap_treshold
            in multiple_overlap_treshold_list)

        self.overlap_thresholds = sorted({
            *self.single_output_overlap_tresholds,
            *multiple_overlap_threshold_list})

        self.overlap_methods = overlap_methods
        self.overlap_class_indices = overlap_class_indices
        self.overlap_class_names = overlap_class_names
        self.confusion_class_names = confusion_class_names
        self.device = device

    def __call__(
            self, overlap_accumulators, confusion_accumulators,
            overlap_confusion_accumulators, binary_confusion_accumulators):
        classwise_overlap_accumulators = {
            name: accumulator.get_classwise_accumulator(
                self.overlap_class_indices, self.device)
            for name, accumulator
            in overlap_accumulators.items()
            if accumulator.has_classes()
        }

        classwise_accumulator_names = classwise_overlap_accumulators.keys()

        pr_curves = [
            (
                name, overlap_threshold,
                accumulator.get_pr_curve(overlap_threshold, self.device))
            for name, accumulator in overlap_accumulators.items()
            for overlap_threshold in self.overlap_thresholds]

        pr_collections = [
            (
                name, overlap_threshold,
                accumulator.get_pr_collection(
                    overlap_threshold, self.device))
            for name, accumulator in classwise_overlap_accumulators.items()
            for overlap_threshold in self.overlap_thresholds]

        classwise_average_precisions = {
            (name, overlap_threshold, method):
            EvaluationHelper._get_ap_list(pr_collection, method)
            for name, overlap_threshold, pr_collection in pr_collections
            for method in self.overlap_methods
        }

        mean_average_precisions_raw = {
            (name, overlap_threshold, method): np.nanmean(ap_list)
            for (name, overlap_threshold, method), ap_list
            in classwise_average_precisions.items()
        }

        average_precisions_raw = {
            (name, overlap_threshold, method):
                EvaluationHelper._get_ap(pr_curve, method)
            for name, overlap_threshold, pr_curve in pr_curves
            for method in self.overlap_methods
        }

        simple_average_precisions = {
            EvaluationHelper._get_name(name, 'AP', overlap_threshold, method):
                average_precisions_raw[name, overlap_threshold, method]
            for name in overlap_accumulators.keys()
            for overlap_threshold in self.single_output_overlap_tresholds
            for method in self.overlap_methods
        }

        simple_mean_average_precisions = {
            EvaluationHelper._get_name(name, 'mAP', overlap_threshold, method):
                mean_average_precisions_raw[name, overlap_threshold, method]
            for name in classwise_overlap_accumulators.keys()
            for overlap_threshold in self.single_output_overlap_tresholds
            for method in self.overlap_methods
        }

        averaged_average_precisions = {
            EvaluationHelper._get_name(
                name, 'AP', multiple_threshold_name, method):
            np.nanmean([
                average_precisions_raw[
                    name, overlap_threshold, method]
                for overlap_threshold
                in overlap_threshold_list])
            for name in overlap_accumulators.keys()
            for multiple_threshold_name, overlap_threshold_list
            in self.multiple_overlap_threshold_lists
            for method in self.overlap_methods
        }

        averaged_mean_average_precisions = {
            EvaluationHelper._get_name(
                name, 'mAP', multiple_threshold_name, method):
            np.nanmean([
                mean_average_precisions_raw[
                    name, overlap_threshold, method]
                for overlap_threshold
                in overlap_threshold_list])
            for name in classwise_overlap_accumulators.keys()
            for multiple_threshold_name, overlap_threshold_list
            in self.multiple_overlap_threshold_lists
            for method in self.overlap_methods
        }

        combined_precision_metrics = {
            **simple_average_precisions, **averaged_average_precisions,
            **simple_mean_average_precisions,
            **averaged_mean_average_precisions}

        single_class_precision_metrics = {
            EvaluationHelper._get_name(
                name, 'class_AP', overlap_threshold, method): {
                    self.overlap_class_names[class_index]: ap
                    for class_index, ap
                    in enumerate(classwise_average_precisions[
                        name, overlap_threshold, method])}
            for name in classwise_overlap_accumulators.keys()
            for overlap_threshold in self.single_output_overlap_tresholds
            for method in self.overlap_methods
        }

        raw_curves = {
            f'{name}_AP_{overlap_threshold}': pr_curve.get_raw_curve()
            for name, overlap_threshold, pr_curve in pr_curves
        }

        confusion_matrices = {
            name: accumulator.get_confusion_matrix(device=self.device)
            for name, accumulator in confusion_accumulators.items()}

        overlap_confusion_matrices = {
            name: accumulator.get_confusion_matrix(device=self.device)
            for name, accumulator in overlap_confusion_accumulators.items()}

        combined_confusion_metrics = {
            f'{name}_avg_iou': confusion_matrix.average_iou
            for name, confusion_matrix
            in chain(
                confusion_matrices.items(),
                overlap_confusion_matrices.items())}

        single_class_normal_confusion_metrics = {
            f'{name}_iou': {
                f'{self.confusion_class_names[index]}': iou
                for index, iou in enumerate(confusion_matrix.iou)}
            for name, confusion_matrix
            in confusion_matrices.items()}

        single_class_overlap_confusion_metrics = {
            f'{name}_iou': {
                f'{self.overlap_class_names[index]}': iou
                for index, iou in enumerate(confusion_matrix.iou)}
            for name, confusion_matrix
            in overlap_confusion_matrices.items()}

        single_class_confusion_metrics = {
            **single_class_normal_confusion_metrics,
            **single_class_overlap_confusion_metrics}

        binary_confusion_matrices = {
            name: accumulator.get_binary_confusion_matrix_collection(
                self.overlap_class_indices)
            for name, accumulator
            in binary_confusion_accumulators.items()}

        confusion_mean_avg_iou = {
            f'{name}_mean_avg_iou':
                confusion_matrix_collection.mean_average_iou
            for name, confusion_matrix_collection
            in binary_confusion_matrices.items()}

        confusion_avg_iou = {
            f'{name}_average_iou':
                confusion_matrix_collection.average_iou
            for name, confusion_matrix_collection
            in binary_confusion_matrices.items()}

        combined_binary_confusion_metrics = {
            **confusion_mean_avg_iou, **confusion_avg_iou}

        single_class_binary_confusion_metrics = {
            f'{name}_iou': {
                f'{self.overlap_class_names[index]}': iou
                for index, iou
                in enumerate(confusion_matrix_collection.classwise_iou)}
            for name, confusion_matrix_collection
            in binary_confusion_matrices.items()}

        combined_metrics = {
            **combined_precision_metrics, **combined_confusion_metrics,
            **combined_binary_confusion_metrics}

        single_class_metrics = {
            **single_class_precision_metrics, **single_class_confusion_metrics,
            **single_class_binary_confusion_metrics}

        return (
            combined_metrics, single_class_metrics,
            raw_curves, confusion_matrices, overlap_confusion_matrices,
            binary_confusion_matrices)


class BinaryMaskConfusionCalculator():
    def __init__(self, mask_threshold):
        super().__init__()
        self.mask_threshold = mask_threshold

    def __call__(self, pred, gt_mask, bbox):
        bool_pred = pred > self.mask_threshold
        return mask_confusion_pair(bool_pred, gt_mask)


class BinaryConfusionAccumulator():
    def __init__(self, binary_confusion_calculator, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.binary_confusion_calculator = binary_confusion_calculator
        self.confusion_matrix_list = list()
        self.label_array_list = list()

    def add_batch(
            self, pred_list_list, gt_mask_list_list, bbox_list_list,
            label_list_list):
        for pred_list, gt_mask_list, bbox_list, label_list in zip(
                pred_list_list, gt_mask_list_list, bbox_list_list,
                label_list_list):
            self.add_sample(pred_list, gt_mask_list, bbox_list, label_list)

    def add_sample(self, pred_list, gt_mask_list, bbox_list, label_list):
        self.label_array_list.append(label_list)
        for pred, gt_mask, bbox in zip(pred_list, gt_mask_list, bbox_list):
            self.confusion_matrix_list.append(
                self.binary_confusion_calculator(pred, gt_mask, bbox))

    def get_binary_confusion_matrix_collection(self, classes, device=None):
        confusion_matrix = torch.stack(self.confusion_matrix_list).to(device)
        labels = torch.cat(self.label_array_list).to(device)
        return BinaryConfusionMatrixCollection(
            confusion_matrix, labels, classes)


class ConfusionCalculator():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

    def __call__(self, pred, gt):
        valid = (0 <= gt) & (gt < self.num_classes)
        selected_pred = pred[valid]
        selected_gt = gt[valid]
        assert (
            (0 <= selected_pred) & (selected_pred < self.num_classes)).all()

        intermediate_number = selected_pred * self.num_classes + selected_gt
        bincount = torch.bincount(
            intermediate_number, minlength=self.num_classes*self.num_classes)
        confusion = bincount.reshape((self.num_classes, self.num_classes))
        return confusion


class ConfusionAccumulator():
    def __init__(self, confusion_calculator, device='cpu'):
        super().__init__()
        self.confusion_calculator = confusion_calculator
        self.device = torch.device(device)
        self.confusion_list = list()

    def add_batch(self, pred, gt):
        result = self.confusion_calculator(pred, gt)
        self.confusion_list.append(result.to(self.device))
    
    def add_list_batch(self, pred, gt):
        self.add_batch(torch.cat(pred), torch.cat(gt))

    def get_confusion_matrix(self, device=None):
        confusion_matrix = torch.stack(self.confusion_list).to(device).sum(0)
        return ConfusionMatrix(confusion_matrix)


class ConfusionMatrix():
    def __init__(self, confusion_matrix):
        tp = torch.diag(confusion_matrix)
        fp = confusion_matrix.sum(1) - tp
        fn = confusion_matrix.sum(0) - tp
        union = tp + fp + fn
        iou = tp.float() / union.float()
        self.confusion_matrix = confusion_matrix.cpu().numpy()
        self.tp = tp.cpu().numpy()
        self.fp = fp.cpu().numpy()
        self.fn = fn.cpu().numpy()
        self.iou = iou.cpu().numpy()
        self.average_iou = np.nanmean(self.iou)


class BinaryConfusionMatrixCollection():
    def __init__(self, confusion_tensor, label_array, classes):
        super().__init__()
        single_class_confusion_tensors = [
            confusion_tensor[label_array == single_class]
            for single_class in classes]

        self.classwise_confusion_matrices = torch.stack([
            single_class_confusion_tensor.sum(0)
            for single_class_confusion_tensor
            in single_class_confusion_tensors], -1).cpu().numpy()

        permuted_single_class_confusion_tensors = (
            single_class_confusion_tensor.permute(1, 2, 0)
            for single_class_confusion_tensor
            in single_class_confusion_tensors)

        single_class_iou_array_list = [
            tp_array.float() / (tp_array + fp_array + fn_array).float()
            for ((tp_array, fp_array), (fn_array, tn_array))
            in permuted_single_class_confusion_tensors]

        self.classwise_mean_iou = np.array([
            iou_array.mean().cpu().item() if len(iou_array) else float('nan')
            for iou_array in single_class_iou_array_list])

        (
            (self.classwise_tp, self.classwise_fp),
            (self.classwise_fn, self.classwise_tn)
        ) = self.classwise_confusion_matrices

        classwise_union = (
            self.classwise_tp + self.classwise_fp + self.classwise_fn)

        self.classwise_iou = np.array([
            class_intersection / class_union if class_union else float('nan')
            for class_intersection, class_union
            in zip(self.classwise_tp, classwise_union)])

        self.average_iou = np.nanmean(self.classwise_iou)
        self.mean_average_iou = np.nanmean(self.classwise_mean_iou)
