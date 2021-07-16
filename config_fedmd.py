global cfg
from config_base import cfg

cfg.update({
    'variant': 'fedmd',
    'alignment_data': 'public',
    'alignment_size': 2000,
    'alignment_target': 'logits',
    'alignment_distillation_loss': 'L1',
    'alignment_matching_epochs': 8,
    'alignment_matching_batch_size': 64,
    'alignment_temperature': 1,
    'private_training_epochs': 2,
})

# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))
