from config_base import cfg

global cfg
cfg.update({
    'variant': 'fedcon',
    'alignment_data': 'public',
    'alignment_size': 200,
    'alignment_target': 'both',
    'alignment_distillation_loss': None,
    'alignment_contrastive_loss': 'contrastive',
    'contrastive_loss_temperature': 1,
    'contrastive_loss_weight': 1,
    'alignment_matching_epochs': 7,
    'alignment_matching_batch_size': 32,
    'alignment_temperature': 1,
    'private_training_epochs': 3,
})
