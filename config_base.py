global cfg
from itertools import islice, cycle, repeat

cfg = {
    'parties': 10,
    'collab_rounds': 100,
    'stages': ['init_public', 'init_private', 'collab', 'lower', 'upper'],

    'projection_head': 256,

    'dataset': 'CIFAR10',
    'classes': None,
    'samples': None,
    'concentration': 0.5,
    'partition_normalize': 'class',

    'private_training_epochs': 10,
    'init_private_batch_size': 64,

    'upper_bound_epochs': 300,
    'lower_bound_epochs': 300,

    'pool_size': 4,
}

cfg['model_mapping'] = list(repeat(3, cfg['parties']))

# import FedMD
# cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
#                                     cfg['parties']))
