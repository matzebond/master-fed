from itertools import islice, cycle, repeat

global cfg
cfg = {
    'parties': 10,
    'collab_rounds': 100,
    'stages': ['init_public', 'init_private', 'collab', 'lower', 'upper'],

    'projection_head': 256,

    'dataset': 'CIFAR10',
    'classes': None,
    'samples_per_class': None,
    'concentration': 0.5,

    'private_training_epochs': 10,

    'pool_size': 4,
}

cfg['model_mapping'] = list(repeat(3, cfg['parties']))

# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))
