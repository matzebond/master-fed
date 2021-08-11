from pathlib import Path
import torch, torch.utils.data
import numpy as np
import umap
import matplotlib.pyplot as plt
import wandb
import wandb_util

api = wandb.Api()
def reps_from_models(models, data, max_size=None, batch_size=64):
    if not isinstance(models, list):
        models = [models]
    dl = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    all_reps = []
    for i, model in enumerate(models):
        reps = []
        for x, _ in dl:
            rep = model(x.to(next(model.parameters()).device), output="rep_only")
            reps.append(rep)
            if max_size and len(reps)*batch_size > max_size:
                break
        reps = torch.cat(reps).detach_()
        all_reps.append(reps)
    labels = torch.tensor(data.targets)[:all_reps[0].size(0)]
    return labels, *all_reps


def create_umap_embedings(reps, reducer=None):
    if reps.shape[-1] > 3:
        if not reducer:
            reducer = umap.UMAP()
        embed = reducer.fit_transform(reps)
    else:
        embed = reps
    return embed, reducer


def build_scatter(embeds, labels, fig=None, spec=[], cmap='jet', size=1):
    if fig is None: fig = plt.figure()
    if spec: spec = [spec]

    ax_args = {'xticks': [], 'yticks': []}
    if embeds.shape[-1] == 3:
        ax_args['projection'] = '3d'
        ax_args['zticks'] = []
    ax = fig.add_subplot(*spec , **ax_args)
    sc = ax.scatter(*embeds.T, s=size, c=labels, cmap=cmap)
    return ax, sc, fig


def build_colorlegend(fig, classes, map=None, cmap='jet'):
    if map is None:
        map = plt.cm.ScalarMappable(norm=None, cmap=cmap)
    num_classes = len(classes)
    cbar = fig.colorbar(map, ax=fig.get_axes(), boundaries=np.arange(num_classes+1)-0.5)
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_ticklabels(classes)
    return cbar
