from itertools import islice, cycle

global cfg
with open('config_base.py') as f:
    exec(f.read())

cfg.update({
    'projection_head': 256,

    'model_averaging': False,
    'keep_prev_model': False,
    'send_global': False,
    'contrastive_loss': 'none',

    'samples_per_class': 20,
    'dataset': 'CIFAR100',
    'concentration': 1,
    'subclasses': [0,2,20,63,71,82],
    'parties': 10,
    'optim': 'Adam',
    'init_public_lr': 0.001,
    'init_public_epochs': 0,
    'init_public_batch_size': 128,
    'init_private_epochs': 0,
    'init_private_batch_size': 32,
    'collab_rounds': 100,
    'alignment_mode': 'public',
    'num_alignment': 1000,
    'logits_matching_epochs': 3,
    'logits_matching_batchsize': 256,
    'logits_temperature': 1,
    'private_training_epochs': 1,
    # 'private_training_batchsize': 5, # TODO not supported
    'upper_bound_epochs': 50,
    'lower_bound_epochs': 50,
})

# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))

cfg['model_mapping'] = list(repeat(3, cfg['parties']))
