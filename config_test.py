from itertools import repeat

from config_base import cfg

global cfg
cfg.update({
    'parties': 2,
    'collab_rounds': 5,
    'stages': ['init_public', 'init_private', 'collab'],

    'projection_head': [256],

    'samples_per_class': 3,
    'dataset': 'CIFAR100',
    'classes': [0,2,20,63,71,82],
    'concentration': 1,

    'variant': None,
    'keep_prev_model': False,
    'global_model': None,
    'replace_local_model': False,
    'send_global': False,
    'contrastive_loss': None,
    'contrastive_loss_weight': 100,
    'contrastive_loss_temperature': 1,
    'alignment_data': 'public',
    'alignment_target': 'both',
    'alignment_distillation_loss': 'L1',
    'alignment_contrastive_loss': 'contrastive+distillation',
    'alignment_size': 128,
    'alignment_matching_epochs': 2,
    'alignment_matching_batch_size': 32,
    'alignment_temperature': 1,

    'private_training_epochs': 1,
    'optim': 'Adam',
    'init_public_lr': 0.001,
    'init_public_epochs': 0,
    'init_public_batch_size': 32,
    'init_private_epochs': 0,
    'init_private_batch_size': 32,
    # 'private_training_batchsize': 5, # TODO not supported
    'upper_bound_epochs': 10,
    'lower_bound_epochs': 10,

    'pool_size': 1,
})

cfg['model_mapping'] = list(repeat(3, cfg['parties']))
