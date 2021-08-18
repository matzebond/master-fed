global cfg
from config_kd import cfg

cfg.update({
    'stages': ['load_global_init_public', 'init_public_from_global', 'init_private', 'collab', 'save_collab'],

    'model_variant': 'LeNet_plus_plus',
    'global_model_mapping': 0,
    # 'model_mapping': 1,

    'dataset': 'MNIST',
    'augmentation': False,

    'alignment_additional_loss': None,
    'alignment_additional_loss_weight': 1,
    'locality_preserving_k': 5,

    'alignment_temperature': 0.5,
    'alignment_distillation_target': None,
    'alignment_distillation_loss': None,
    'alignment_distillation_weight': 2,

    'alignment_label_loss': False,
    'alignment_label_loss_weight': 1,

    'alignment_matching_batch_size': 256,

    # 'optim_lr': 0.001,
    # 'optim_weight_decay': 0.9,
})

cfg.update({
    'model_mapping': cfg['global_model_mapping']
})
