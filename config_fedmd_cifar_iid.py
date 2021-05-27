from itertools import islice, cycle
import FedMD

global cfg
with open('config_base.py') as f:
    exec(f.read())

cfg.update({
    'samples_per_class': 3,
    'dataset': 'CIFAR100',
    'concentration': 'iid',
    'subclasses': [0,2,20,63,71,82],
    'parties': 10,
    'optim': 'Adam',
    'init_public_lr': 0.001,
    'init_public_epochs': 20,
    'init_public_batch_size': 128,
    'init_private_epochs': 25,
    'init_private_batch_size': 32,
    'collab_rounds': 20,
    'alignment_mode': 'public',
    'num_alignment': 5000,
    'logits_matching_epochs': 1,
    'logits_matching_batchsize': 256,
    'logits_temperature': 1,
    'private_training_epochs': 4,
    # 'private_training_batchsize': 5, # TODO not supported
    'model_averaging': False,
})

cfg['model_mapping'] =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR.hyper))),
                                    cfg['parties']))
