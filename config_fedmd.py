from config_base import cfg

global cfg
cfg.update({
    'variant': 'fedmd',
    'alignment_data': 'public',
    'alignment_size': 200,
    'alignment_target': 'logits',
    'alignment_distillation_loss': 'L1',
    'alignment_matching_epochs': 7,
    'alignment_matching_batchsize': 32,
    'alignment_temperature': 1,
    'private_training_epochs': 3,
})

# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))
