from itertools import repeat

global cfg
with open('config_base.py') as f:
    exec(f.read())

cfg.update({
    'stages': ['init_public', 'init_private', 'collab'],
    'pool_size': 2,

    'model_averaging': True,
    'samples_per_class': 20,
    'dataset': 'CIFAR100',
    'concentration': 0.1,
    'subclasses': [0,2,20,63,71,82],
    'parties': 4,
    'optim': 'Adam',
    'init_public_lr': 0.001,
    'init_public_epochs': 0,
    'init_public_batch_size': 128,
    'init_private_epochs': 0,
    'init_private_batch_size': 32,
    'collab_rounds': 5,
    'alignment_mode': 'none',
    'num_alignment': 100,
    'logits_matching_epochs': 2,
    'logits_matching_batchsize': 256,
    'logits_temperature': 1,
    'private_training_epochs': 2,
    # 'private_training_batchsize': 5, # TODO not supported
})

cfg['model_mapping'] = list(repeat(3, cfg['parties']))
