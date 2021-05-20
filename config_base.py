import wandb

global cfg
cfg = {
    'group': wandb.util.generate_id(),
    'stages': ['init_public', 'init_private', 'collab', 'lower', 'upper'],
    'pool_size': 4,
    'ignore': ['ignore', 'group', 'rank', 'model'], #'pool_size'],
}
