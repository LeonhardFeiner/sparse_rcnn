# %%
import torch
from torch.utils.tensorboard import SummaryWriter

import shutil
import os
from ndsis.training.training import train
from ndsis.training.kill_switch import KillSwitch
from scannet_config.run import (
    debug, log_storage, model_storage, experiment, complete_config,
    model, loss, optimizer, scheduler, device,
    batch_to_device, get_data_loader, train_loader_params, val_loader_params,
    selected_train_path, selected_val_path, selected_trainval_path,
    flat_config, param_configs, pre_augmented,
    selected_train_scene_ids, selected_val_scene_ids,
    bbox_overlap_calculator, mask_overlap_calculator,
    mask_confusion_calculator, label_confusion_calculator, evaluation_helper,
    evaluate_every_epoch, evaluate_every_step, batches_per_step,
    scene_count_start, continue_training, epochs, error_threshold,
    confusion_calculator, do_eval)


# %%
if not debug:
    assert len(experiment) <= 256, (len(experiment), experiment)
print('starting', experiment)

train_loader = get_data_loader(
    selected_train_scene_ids, selected_train_path,
    **train_loader_params)
val_loader = get_data_loader(
    selected_val_scene_ids, selected_val_path,
    **val_loader_params)

trainval_loader_params = val_loader_params.copy()
trainval_loader_params['data_slice'] = slice(len(val_loader.dataset))

trainval_loader = get_data_loader(
    selected_train_scene_ids, selected_trainval_path,
    **trainval_loader_params)

if pre_augmented:
    def set_epoch(epoch):
        train_loader.dataset.set_epoch(epoch)
        trainval_loader.dataset.set_epoch(epoch)
        val_loader.dataset.set_epoch(epoch)
else:
    def set_epoch(epoch):
        pass

# %%

if debug:
    checkpoint_storage = None
    loss_checkpoint_storage = None
    experiment = 'debug'
    tensorboard_dir = f'/tmp/tensorboardlog/{experiment}/'
    call_graph_storage = 'pycallgraph.png'
else:
    full_model_storage = model_storage + experiment + '/'
    extention = 1
    orig_experiment = experiment
    while(os.path.exists(full_model_storage) and not continue_training):
        print(full_model_storage, 'already exists!')
        extention += 1
        experiment = f'{orig_experiment}_trial{extention}'
        full_model_storage = model_storage + experiment + '/'

    code_dir = full_model_storage + 'code/'
    tensorboard_dir = full_model_storage + 'log/'
    checkpoint_storage = full_model_storage + 'checkpoint/'
    loss_checkpoint_storage = full_model_storage + 'loss_checkpoint/'
    call_graph_storage = full_model_storage + '/pycallgraph.png'

    if not continue_training:
        os.makedirs(checkpoint_storage)
        os.mkdir(loss_checkpoint_storage)

        ignore = (
            '__pycache__', '*.pyc', '.gitignore', 'LICENSE', 'preparation',
            'test', '.git', '*.md', '.vscode', 'environment.yml', '*.ply')

        try:
            shutil.copytree(
                '.',
                code_dir,
                ignore=shutil.ignore_patterns(*ignore))
            os.symlink(tensorboard_dir, log_storage + experiment)
        except (shutil.Error, OSError) as e:
            print(e)

kill_switch = KillSwitch(min(epochs), max(epochs)+1)

train_writer = SummaryWriter(
    tensorboard_dir,
    purge_step=(scene_count_start if scene_count_start else None))
train_val_writer = None
val_writer = None

# %%
train_writer.add_text('config', str(complete_config), global_step=min(epochs))
for key, value in param_configs.items():
    train_writer.add_text('config/' + key, str(value), global_step=min(epochs))
# %%

def get_multiline(name, plot_names, section_names):
    inner_layout = {
        plot_name: ['Multiline', [
            f'{section_name}_{name}/{plot_name}'
            for section_name in section_names]]
        for plot_name in plot_names}

    return name, inner_layout

def get_layout(dict_dict, section_names):
    return dict(
        get_multiline(name, plot_names, section_names)
        for name, plot_names in dict_dict.items())


loss_names = ['_total_', 'class', 'mask', 'rpn_bbox', 'rpn_score']
metric_names = [
    'bbox_AP_0.25', 'bbox_AP_0.5', 'bbox_mAP_0.25', 'bbox_mAP_0.5',
    'mask_AP_0.25', 'mask_AP_0.5', 'mask_mAP_0.25', 'mask_mAP_0.5',
    'segment_avg_iou']
section_names = ['trainval', 'val']
layout = get_layout(
    {'loss': loss_names, 'metric': metric_names}, section_names)


# %%

train_writer.add_custom_scalars(layout)

val_losses, val_metrics, train_losses, train_metrics = train(
    model, loss, train_loader, val_loader, trainval_loader,
    optimizer, scheduler, device, batch_to_device, experiment,
    checkpoint_storage, loss_checkpoint_storage,
    train_writer, val_writer, train_val_writer,
    bbox_overlap_calculator, mask_overlap_calculator,
    confusion_calculator, mask_confusion_calculator,
    label_confusion_calculator, evaluation_helper,
    evaluate_every_step, evaluate_every_epoch, epochs, kill_switch,
    error_threshold, batches_per_step, set_epoch,
    scene_counter=scene_count_start, do_eval=do_eval)

train_metic_dict = {
    f'train_summary/{metric}': train_metrics.get(metric, 0)
    for metric in metric_names}
val_metric_dict = {
    f'val_summary/{metric}': val_metrics.get(metric, 0)
    for metric in metric_names}
metric_dict = {**val_metric_dict, **train_metic_dict}

print(flat_config)
print(metric_dict)
train_writer.add_hparams(hparam_dict=flat_config, metric_dict=metric_dict)

# %%
