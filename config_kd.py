global cfg

cfg = {
    'parties': 1,
    'collab_rounds': 100,
    'stages': ['load_global_init_public', 'init_public', 'init_private', 'collab'],

    'model_variant': 'LPP', #'LeNet_plus_plus',
    'global_model_mapping': 4,
    'global_projection_head': 500,
    'model_mapping': 1,
    'projection_head': 500,

    'dataset': 'CIFAR10',
    'public_dataset': 'same',
    'classes': None,
    'samples': None,
    'concentration': 'iid',
    'partition_normalize': 'class',
    # 'partition_normalize': 'party',
    # 'samples': 5000,

    'variant': None,
    'global_model': 'fix',
    'replace_local_model': False,
    'keep_prev_model': False,
    'send_global': False,

    'global_init_public_epochs': 100,

    'alignment_data': 'public',
    'alignment_size': 'full',
    'alignment_aggregate': 'global',
    'alignment_target': 'both',
    'alignment_temperature': 1,
    'alignment_additional_loss': 'locality_preserving',
    'alignment_additional_loss_weight': 1,
    'locality_preserving_k': 5,
    'alignment_distillation_target': 'logits',
    'alignment_distillation_loss': 'MSE',
    'alignment_distillation_weight': 1,
    'alignment_label_loss': True,
    'alignment_label_loss_weight': 1,
    'alignment_matching_epochs': 1,

    'private_training_epochs': 0,
    'init_public_batch_size': 32,
    'init_private_batch_size': 32,
    'alignment_matching_batch_size': 128,
    # 'optim_lr': 0.001,
    # 'optim_weight_decay': 0.9,

    # 'upper_bound_epochs': 100,
    'lower_bound_epochs': 100,

    'pool_size': 1,
}

# from itertools import islice, cycle, repeat
# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))
