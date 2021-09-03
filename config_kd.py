global cfg

cfg = {
    'parties': 1,
    'collab_rounds': 1,
    'stages': ['load_global_init_public', 'init_via_global', 'init_public', 'init_private', 'collab', 'save_collab'],

    'model_variant': 'LPP',
    'global_model_mapping': 4,
    'global_projection_head': 500,
    'model_mapping': 1,
    'projection_head': 500,

    'dataset': 'CIFAR10',
    'classes': None,
    'samples': None,
    'concentration': 'iid',
    'partition_normalize': 'class',
    # 'partition_normalize': 'party',
    # 'samples': 5000,
    'public_dataset': 'CIFAR10',

    'variant': None,
    'global_model': 'fix',
    'replace_local_model': False,
    'keep_prev_model': False,
    'send_global': False,

    'global_init_public_epochs': 100,

    'alignment_data': 'private',
    'alignment_size': 'full',
    'alignment_aggregate': 'global',
    'alignment_target': 'both',
    'alignment_temperature': 0.5,
    'alignment_additional_loss': None,
    'alignment_additional_loss_weight': 1,
    'locality_preserving_k': 5,
    'alignment_distillation_target': None,
    'alignment_distillation_loss': None,
    'alignment_distillation_weight': 2,
    'alignment_label_loss': False,
    'alignment_label_loss_weight': 1,
    'alignment_matching_epochs': 100,

    'private_training_epochs': 0,
    'init_public_batch_size': 32,
    'init_private_batch_size': 32,
    'alignment_matching_batch_size': 256,

    # 'optim_lr': 0.001,
    # 'optim_weight_decay': 0.9,

    # 'upper_bound_epochs': 100,
    # 'lower_bound_epochs': 100,

    'pool_size': 1,
}

# from itertools import islice, cycle, repeat
# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))
