import numpy as np
import torch
from tqdm import tqdm
import gc
from functools import partial
from collections import defaultdict
from ndsis.training.evaluation import (
    OverlapAccumulator, EvaluationHelper, ConfusionAccumulator,
    BinaryConfusionAccumulator)


tqdm_cols = 200


def combine_dict(the_dict, precision):
    return ' '.join(
        f'{key}={value:{precision}}' for key, value in the_dict.items())


def write_loss(epoch, step, scenes_count, description, losses, metrics={}):
    tqdm.write(
        f"{description:<8} "
        f"epoch {epoch:>3} step {step:>4} scenes_count {scenes_count:>7}")
    tqdm.write(
        f"{description:<8} "
        f"loss=[{combine_dict(losses, '2.6f')}]")
    tqdm.write(
        f"{description:<8} "
        f"metric=[{combine_dict(metrics, '1.4f')}]")


def write_tensorboard_loss(
        full_step, writers, losses,
        all_class_metrics=None, single_class_metrics=None, pr_curves=None):
    for graph_name, writer in writers.items():
        for loss_name, loss_value in losses.items():
            writer.add_scalar(
                f'{graph_name}_loss/{loss_name}',
                loss_value, full_step)

    if all_class_metrics is not None:
        for graph_name, writer in writers.items():
            for key, value in all_class_metrics.items():
                writer.add_scalar(
                    f'{graph_name}_metric/{key}',
                    value, full_step)

    if single_class_metrics is not None:
        for graph_name, writer in writers.items():
            for metric_name, class_dict in single_class_metrics.items():
                for key, value in class_dict.items():
                    writer.add_scalar(
                        f'{graph_name}_metric_{metric_name}/{key}',
                        value, full_step)

    if pr_curves is not None:
        for graph_name, writer in writers.items():
            for key, (tp, fp, tn, fn, precision, recall, num_samples) in \
                    pr_curves.items():
                writer.add_pr_curve_raw(
                    f'{graph_name}/{key}', tp, fp, tn, fn, precision, recall,
                    global_step=full_step, num_thresholds=num_samples)


def write_params(index, writers, **params):
    for writer in writers.values():
        for name, value in params.items():
            writer.add_scalar(f'_param/{name}', value, index)


def write_weights(index, writers, **params):
    for writer in writers.values():
        for name, value in params.items():
            writer.add_scalar(f'weights/{name}', value, index)


def check_kill(kill_switch, epoch, step, steps_per_epoch, model, loss, device):
    if kill_switch and not kill_switch.write_and_check(
            epoch, step, steps_per_epoch):
        tqdm.write('#### Kill Switch triggered! ####')
        model.cpu()
        loss.cpu()
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        tqdm.write('Please enter "kill" or "continue".')
        answer = None
        while answer != "continue":
            answer = input('\nEnter "kill" or "continue": ')
            if answer == "kill":
                tqdm.write("Process is Killed!")
                exit(-1)
        model.to(device)
        loss.to(device)
        tqdm.write("Continue Process")
            

class AccumulatorCollection():
    def __init__(
            self, evaluation_helper, model, bbox_overlap_calculator,
            mask_overlap_calculator, segmentation_confusion_calculator,
            mask_binary_confusion_calculator, label_confusion_calculator,
            eval_on_gt=True):
        super().__init__()
        self.evaluation_helper = evaluation_helper
        self.evaluate_bbox = model.calc_bbox
        self.evaluate_gt_bbox = model.calc_classes and eval_on_gt
        self.evaluate_masks = (
            model.calc_masks and model.calc_classes and model.calc_bbox)
        self.evaluate_gt_bbox_mask = (
            model.calc_masks and model.calc_classes and eval_on_gt)
        self.evaluate_gt_bbox_gt_label_mask = (
            model.calc_masks and eval_on_gt)
        self.evaluate_segmentation = model.calc_segmentation
        if self.evaluate_bbox:
            self.bbox_overlap_accumulator = OverlapAccumulator(
                bbox_overlap_calculator)
        if self.evaluate_gt_bbox:
            self.gt_bbox_overlap_accumulator = OverlapAccumulator(
                bbox_overlap_calculator)
            self.gt_bbox_confusion_accumulator = ConfusionAccumulator(
                label_confusion_calculator)
        if self.evaluate_masks:
            self.mask_overlap_accumulator = OverlapAccumulator(
                mask_overlap_calculator)
        if self.evaluate_gt_bbox_mask:
            self.gt_mask_overlap_accumulator = OverlapAccumulator(
                mask_overlap_calculator)
        if self.evaluate_gt_bbox_gt_label_mask:
            self.gt_label_mask_overlap_accumulator = OverlapAccumulator(
                mask_overlap_calculator)
            self.gt_label_mask_confusion_accumulator = \
                BinaryConfusionAccumulator(mask_binary_confusion_calculator)
        if self.evaluate_segmentation:
            self.segmentation_accumulator = ConfusionAccumulator(
                segmentation_confusion_calculator)

    def get_metrics(self):
        overlap_accumulator_dict = dict()
        confusion_accumulator_dict = dict()
        mask_confusion_accumulator_dict = dict()
        binary_confusion_accumulator_dict = dict()
        if self.evaluate_bbox:
            overlap_accumulator_dict['bbox'] = \
                self.bbox_overlap_accumulator
        if self.evaluate_gt_bbox:
            overlap_accumulator_dict['gtbbox'] = \
                self.gt_bbox_overlap_accumulator
            mask_confusion_accumulator_dict['gtbbox'] = \
                self.gt_bbox_confusion_accumulator
        if self.evaluate_masks:
            overlap_accumulator_dict['mask'] = \
                self.mask_overlap_accumulator
        if self.evaluate_gt_bbox_mask:
            overlap_accumulator_dict['gtmask'] = \
                self.gt_mask_overlap_accumulator
        if self.evaluate_gt_bbox_gt_label_mask:
            overlap_accumulator_dict['gtlabelmask'] = \
                self.gt_label_mask_overlap_accumulator
            binary_confusion_accumulator_dict['gtlabelmask'] = \
                self.gt_label_mask_confusion_accumulator
        if self.evaluate_segmentation:
            confusion_accumulator_dict['segment'] = \
                self.segmentation_accumulator

        (
            combined_metrics, single_class_metrics,
            raw_pr_curves, raw_confusion_matrices, overlap_confusion_matrices,
            binary_confusion_matrices
        ) = self.evaluation_helper(
            overlap_accumulator_dict, confusion_accumulator_dict, 
            mask_confusion_accumulator_dict, binary_confusion_accumulator_dict)

        combined_metrics = {
            key: value
            for key, value in combined_metrics.items()
            if 'gtbbox_AP' not in key}

        return (
            combined_metrics, single_class_metrics,
            raw_pr_curves, raw_confusion_matrices, overlap_confusion_matrices,
            binary_confusion_matrices)

    def add_batch(self, batch, outputs):
        pseudo_score = [
            gt_bbox.new_ones((len(gt_bbox),))
            for gt_bbox in batch['gt_bbox']]

        if self.evaluate_bbox:
            if 'class' in outputs:
                self.bbox_overlap_accumulator.add_batch(
                    outputs['roi_score'],
                    outputs['roi_bbox'],
                    batch['gt_bbox'],
                    outputs['class'],
                    batch['gt_label'])
            else:
                self.bbox_overlap_accumulator.add_batch(
                    outputs['roi_score'],
                    outputs['roi_bbox'],
                    batch['gt_bbox'])
        if self.evaluate_gt_bbox:
            self.gt_bbox_overlap_accumulator.add_batch(
                pseudo_score,
                batch['gt_bbox'],
                batch['gt_bbox'],
                outputs['gt_bbox_class'],
                batch['gt_label'])
            self.gt_bbox_confusion_accumulator.add_list_batch(
                outputs['gt_bbox_class'],
                batch['gt_label'])
        if self.evaluate_masks:
            self.mask_overlap_accumulator.add_batch(
                outputs['roi_score'],
                outputs['mask'],
                batch['gt_mask'],
                outputs['class'],
                batch['gt_label'])
        if self.evaluate_gt_bbox_mask:
            self.gt_mask_overlap_accumulator.add_batch(
                pseudo_score,
                outputs['gt_bbox_mask'],
                batch['gt_mask'],
                outputs['gt_bbox_class'],
                batch['gt_label'])
        if self.evaluate_gt_bbox_gt_label_mask:
            self.gt_label_mask_overlap_accumulator.add_batch(
                pseudo_score,
                outputs['gt_bbox_gt_label_mask'],
                batch['gt_mask'],
                batch['gt_label'],
                batch['gt_label'])
            self.gt_label_mask_confusion_accumulator.add_batch(
                outputs['gt_bbox_gt_label_mask'],
                batch['gt_mask'],
                batch['gt_bbox'],
                batch['gt_label'])
        if self.evaluate_segmentation:
            self.segmentation_accumulator.add_batch(
                outputs['segmentation_class'],
                batch['gt_segmentation'])


def eval_model(
        model, loss_function, data_loader, batch_to_device, device, writers,
        bbox_overlap_calculator, mask_overlap_calculator, confusion_calculator,
        mask_confusion_calculator, label_confusion_calculator,
        evaluation_helper, epoch, step, scenes_count, description,
        eval_on_gt=True):

    loss_dict = defaultdict(list)
    model.eval()
    if len(data_loader) > 1:
        tqdm_loader = tqdm(
            data_loader, desc=description, leave=False, ncols=tqdm_cols)
        write_postfix = True
    else:
        tqdm_loader = data_loader
        write_postfix = False

    accumulator_collection = AccumulatorCollection(
        evaluation_helper, model, bbox_overlap_calculator,
        mask_overlap_calculator, confusion_calculator,
        mask_confusion_calculator, label_confusion_calculator, eval_on_gt)

    with torch.no_grad():
        for batch in tqdm_loader:
            batch = batch_to_device(batch, device)
            outputs = model(
                batch['data'],
                gt_label=batch['gt_label'],
                gt_bbox=batch['gt_bbox'],
                calc_gt_predictions=eval_on_gt)
            loss, losses = loss_function(batch, outputs)

            accumulator_collection.add_batch(batch, outputs)

            for key, value in losses.items():
                loss_dict[key].append(value)

            if write_postfix:
                tqdm_loader.set_postfix(**losses)

            del outputs, batch, loss

    losses = {key: np.mean(array) for key, array in loss_dict.items()}

    (
        combined_metrics, single_class_metrics,
        raw_pr_curves, raw_confusion_matrices, overlap_confusion_matrices,
        binary_confusion_matrices
    ) = accumulator_collection.get_metrics()

    write_tensorboard_loss(
        scenes_count, writers, losses,
        combined_metrics, single_class_metrics)

    write_loss(
        epoch, step, scenes_count, description, losses=losses,
        metrics=combined_metrics)

    model.train()
    
    return losses, combined_metrics


def train(
        model, loss_function, train_loader, val_loader, train_val_loader,
        optimizer, scheduler,
        device, batch_to_device, experiment, model_storage, loss_storage,
        train_writer, val_writer=None, train_val_writer=None,
        bbox_overlap_calculator=None, mask_overlap_calculator=None, 
        confusion_calculator=None, mask_confusion_calculator=None,
        label_confusion_calculator=None, evaluation_helper=None,
        evaluate_every_step=0, evaluate_every_epoch=0,
        epochs=[0], kill_switch=None, error_threshold=0.1, batches_per_step=1,
        set_epoch=lambda x: None, scene_counter=0, do_eval=True):

    if isinstance(batches_per_step, int):
        batches_per_step_list = [batches_per_step] * (max(epochs) + 1)
    else:
        batches_per_step_list = batches_per_step

    train_writers = dict(train=train_writer)
    val_writers = dict(val=train_writer)
    train_val_writers = dict(trainval=train_writer)
    if val_writer is not None:
        val_writers['train'] = val_writer
    if train_val_writer is not None:
        train_val_writers['train'] = train_val_writer

    train_writer.add_text('model', str(model), min(epochs))
    train_writer.add_text('experiment', experiment, min(epochs))
    train_writer.add_text('epochs', f'from {min(epochs)} to {max(epochs)}')
    steps_per_epoch = len(train_loader)

    if not model_storage:
        basis_path = None
    elif evaluate_every_step:
        basis_path = model_storage + 'epoch{0:0>4}_step{1:0>4}.pth'
    else:
        basis_path = model_storage + 'epoch{0:0>4}.pth'

    if not loss_storage:
        loss_basis_path = None
    elif evaluate_every_step:
        loss_basis_path = loss_storage + 'epoch{0:0>4}_step{1:0>4}.pth'
    else:
        loss_basis_path = loss_storage + 'epoch{0:0>4}.pth'

    model.to(device)
    loss_function.to(device)

    def eval_check(
            epoch, step, scenes_count, train_batch=None):

        tqdm.write(experiment)

        val_losses, val_metrics = eval_model(
            model, loss_function, val_loader, batch_to_device, device,
            val_writers, bbox_overlap_calculator, mask_overlap_calculator,
            confusion_calculator, mask_confusion_calculator,
            label_confusion_calculator, evaluation_helper,
            epoch, step, scenes_count, 'val')

        train_losses, train_metrics = eval_model(
            model, loss_function, train_val_loader, batch_to_device, device,
            train_val_writers, bbox_overlap_calculator,
            mask_overlap_calculator, confusion_calculator,
            mask_confusion_calculator, label_confusion_calculator,
            evaluation_helper,
            epoch, step, scenes_count, 'trainval')

        if train_batch is not None:
            train_losses = eval_model(
                model, loss_function, [train_batch], batch_to_device, device,
                train_writers, bbox_overlap_calculator,
                mask_overlap_calculator, confusion_calculator,
                mask_confusion_calculator, label_confusion_calculator,
                evaluation_helper,
                epoch, step, scenes_count,
                'train')

        model.train()

        if basis_path:
            torch.save(model.state_dict(), basis_path.format(epoch, step))
        if loss_basis_path:
            torch.save(
                loss_function.state_dict(),
                loss_basis_path.format(epoch, step))

        check_kill(
            kill_switch, epoch, step, steps_per_epoch, model, loss_function,
            device)

        return val_losses, val_metrics, train_losses, train_metrics

    print(f'num_train: {len(train_loader)} num_val: {len(val_loader)} '
          f'name: {experiment}')

    epoch_tqdm = tqdm(epochs, desc='epochs', ncols=tqdm_cols)

    error_counter = 0
    for epoch in epoch_tqdm:
        set_epoch(epoch)
        batches_per_step = batches_per_step_list[epoch]
        if len(train_loader) > 1:
            tqdm_train_loader = tqdm(
                train_loader, desc='training', leave=False, ncols=tqdm_cols)
            loss_postfix_writer = tqdm_train_loader.set_postfix
        else:
            tqdm_train_loader = train_loader
            loss_postfix_writer = epoch_tqdm.set_postfix
            # partial(
            #     epoch_tqdm.set_postfix, name=experiment)

        for step, batch in enumerate(tqdm_train_loader, 0):
            full_step = epoch * len(train_loader) + step

            if (
                do_eval and
                ((not step and
                (not evaluate_every_epoch or not epoch % evaluate_every_epoch))
                or evaluate_every_step and not step % evaluate_every_step)):
                eval_check(epoch, step, scene_counter, batch)

            batch = batch_to_device(batch, device)
            model.train()
            try:
                outputs = model(batch['data'], gt_bbox=batch['gt_bbox'])
                loss, train_losses = loss_function(batch, outputs)
                if torch.isnan(loss):
                    raise RuntimeError(
                        f'Loss is nan! sublosses:{train_losses}')
                (loss / batches_per_step).backward()
            except RuntimeError as e:
                error_counter += 1
                if error_counter / (full_step + 1) > error_threshold:
                    raise
                error_text = (
                    f"{e}\n"
                    f"{error_counter} out of {full_step} failed "
                    f"(proportion of {error_counter/full_step}).")
                tqdm.write(error_text)
                train_writer.add_text('error', error_text, epoch)
                tqdm.write('pre empty cache')
                tqdm.write(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                tqdm.write('post empty cache')
                tqdm.write(torch.cuda.memory_summary())
            else:
                write_tensorboard_loss(
                    scene_counter, train_writers, train_losses)
                loss_postfix_writer(
                    train_loss=train_losses['_total_'])
            finally:
                if not (step + 1) % batches_per_step:
                    optimizer.step()
                    optimizer.zero_grad()
                scene_counter += len(batch['id'])
                outputs, batch, loss = None, None, None
                del outputs, batch, loss

        if scheduler is None:
            current_lr = optim.defaults['lr']
        else:
            current_lr = scheduler.get_last_lr()[0]
        write_params(
            scene_counter, train_writers,
            step=step, epoch=epoch, full_step=full_step, lr=current_lr)

        write_weights(
            scene_counter, train_writers, **loss_function.get_weights())

        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        #torch.cuda.empty_cache()
        tqdm.write(f'epoch {epoch}')
        #tqdm.write(torch.cuda.memory_summary())

        tqdm_train_loader = None
        check_kill(
            kill_switch, epoch, step, steps_per_epoch, model, loss_function,
            device)

    if do_eval:
        return_value = eval_check(
            epoch + 1, 0, scene_counter, next(iter(train_loader)))

    print(f'{experiment} finished')
    if error_counter:
        print(f"{error_counter} out of {full_step} failed "
              f"(proportion of {error_counter/full_step}).")

    if do_eval:
        return return_value
    else:
        return ({},) * 4
