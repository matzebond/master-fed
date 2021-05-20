from itertools import repeat

global cfg
with open('config_base.py') as f:
    exec(f.read())

cfg.update({
    'model_averaging': True,
    'samples_per_class': 3,
    'dataset': 'CIFAR100',
    'data_variant': 'iid',
    'subclasses': [0,2,20,63,71,82],
    'parties': 10,
    'load_private_idx': True,
    'optim': 'Adam',
    'init_public_lr': 0.001,
    'init_public_epochs': 0,
    'init_public_batch_size': 128,
    'init_private_epochs': 0,
    'init_private_batch_size': 32,
    'collab_rounds': 20,
    'alignment_mode': 'none',
    'num_alignment': 0,
    'logits_matching_epochs': 0,
    'logits_matching_batchsize': 256,
    'private_training_epochs': 5,
    # 'private_training_batchsize': 5, # TODO not supported
})

cfg['model_mapping'] = list(repeat(3, cfg['parties']))
