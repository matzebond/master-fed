from itertools import repeat

global cfg
with open('config_base.py') as f:
    exec(f.read())

cfg.update({
    'variant': 'moon',
    'projection_head': 256,
    'samples_per_class': 20,
    'dataset': 'CIFAR100',
    'concentration': 1,
    'subclasses': [0,2,20,63,71,82],
    'parties': 10,
    'optim': 'Adam',
    'init_public_lr': 0.0001,
    'init_public_epochs': 0,
    'init_public_batch_size': 32,
    'init_private_epochs': 0,
    'init_private_batch_size': 32,
    'collab_rounds': 100,
    'private_training_epochs': 5,
    'contrastive_loss_weight': 5,
    'contrastive_loss_temperature': 1,
    # 'private_training_batchsize': 5, # TODO not supported
    'upper_bound_epochs': 50,
    'lower_bound_epochs': 50,
})

cfg['model_mapping'] = list(repeat(3, cfg['parties']))
