global cfg
from config_kd import cfg

cfg.update({
    'stages': ['load_global_init_public', 'init_public', 'init_private', 'collab', 'save_collab'],

    'model_variant': 'LeNet_plus_plus',
    'global_model_mapping': 0,
    'model_mapping': 1,

    'dataset': 'MNIST',
    'augmentation': False,

    'alignment_additional_loss': 'locality_preserving',
    'alignment_additional_loss_weight': 1,
    'locality_preserving_k': 5,

    'alignment_temperature': 0.5,
    'alignment_distillation_target': 'logits',
    'alignment_distillation_loss': 'KL',
    'alignment_distillation_weight': 2,

    'alignment_label_loss': True,
    'alignment_label_loss_weight': 1,

    'private_training_epochs': 0,
    'init_public_batch_size': 32,
    'init_private_batch_size': 32,
    'alignment_matching_batch_size': 256,
})
