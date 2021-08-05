global cfg
from config_base import cfg

cfg.update({
    'variant': 'fedcon',
    'alignment_data': 'public',
    'alignment_size': 2000,
    'alignment_target': 'both',
    'alignment_distillation_loss': None,
    'alignment_additional_loss': 'contrastive',
    'alignment_additional_loss_weight': 1,
    'contrastive_loss_temperature': 0.5,
    'alignment_matching_epochs': 8,
    'alignment_matching_batch_size': 64,
    'alignment_temperature': 1,
    'private_training_epochs': 2,
})
