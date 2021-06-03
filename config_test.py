from itertools import repeat

global cfg
with open('config_base.py') as f:
    exec(f.read())

cfg.update({
    'stages': ['init_public', 'init_private', 'save_init_private', 'collab', 'save_final'],
    'pool_size': 2,

    'projection_head': 64,

    'variant': None,
    'keep_prev_model': True,
    'global_model': 'averaging',
    'replace_local_model': True,
    'send_global': True,
    'send_rep': False,
    'contrastive_loss': 'moon',
    'contrastive_loss_weight': 5,
    'contrastive_loss_temperature': 1,
    'alignment_mode': 'none',

    'samples_per_class': 20,
    'dataset': 'CIFAR100',
    'concentration': 0.1,
    'subclasses': [0,2,20,63,71,82],
    'parties': 2,
    'optim': 'Adam',
    'init_public_lr': 0.001,
    'init_public_epochs': 0,
    'init_public_batch_size': 32,
    'init_private_epochs': 0,
    'init_private_batch_size': 32,
    'collab_rounds': 5,
    'num_alignment': 100,
    'logits_matching_epochs': 2,
    'logits_matching_batchsize': 32,
    'logits_temperature': 1,
    'private_training_epochs': 1,
    # 'private_training_batchsize': 5, # TODO not supported
    'upper_bound_epochs': 2,
    'lower_bound_epochs': 2,
})

cfg['model_mapping'] = list(repeat(3, cfg['parties']))
