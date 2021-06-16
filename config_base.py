from itertools import islice, cycle, repeat

global cfg
cfg = {
    'parties': 10,
    'collab_rounds': 100,
    'stages': ['init_public', 'init_private', 'collab', 'lower', 'upper'],

    'projection_head': 256,

    'samples_per_class': 20,
    'dataset': 'CIFAR100',
    'concentration': 1,
    'subclasses': [0,2,20,63,71,82],

    'variant': None,
    'private_training_epochs': 10,

    'pool_size': 4,
    'ignore': ['ignore', 'rank', 'model', 'path', 'tmp'], #'pool_size'],
}

cfg['model_mapping'] = list(repeat(3, cfg['parties']))

# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))
