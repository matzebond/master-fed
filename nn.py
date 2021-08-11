import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance as scidist
import copy
from typing import List, Callable, Optional, Dict, Sequence, Union, Tuple, Mapping


def avg_params(models: Sequence[nn.Module]) -> Mapping[str, Tensor]:
    assert len(models) > 0

    global_weights = copy.deepcopy(models[0].state_dict())
    for model in models[1:]:
        model_weights = model.state_dict()
        for key in model_weights:
            global_weights[key] += model_weights[key]

    for key in global_weights:
        global_weights[key] = global_weights[key] / len(models)

    return global_weights


def reset_all_parameters(model):
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()
    else:
        pass


def optim_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return optimizer


class KLDivSoftmaxLoss(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(KLDivSoftmaxLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(F.log_softmax(input, dim=-1), target,
                        reduction=self.reduction, log_target=self.log_target)

# class CrossEntropyDistillation(nn.modules.loss._Loss):
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
#         super(CrossEntropyDistillation, self).__init__(size_average, reduce, reduction)

#     def forward(outputs, targets):
#         log_softmax_outputs = F.log_softmax(outputs/self.temperature, dim=1)
#         softmax_targets = F.softmax(targets/self.temperature, dim=1)
#         return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def moon_contrastive_loss(x, local_rep, global_model, prev_model, temperature, device="cpu"):
    with torch.no_grad():
        global_rep = global_model(x, output='rep_only')
        prev_rep = prev_model(x, output='rep_only')

    pos = F.cosine_similarity(local_rep, global_rep, dim=-1).reshape(-1,1)
    neg = F.cosine_similarity(local_rep, prev_rep, dim=-1).reshape(-1,1)

    logits = torch.cat((pos, neg), dim=1) / temperature

    # first "class" sim(global_rep) is the ground truth
    labels = torch.zeros(x.size(0), device=device).long()
    return F.cross_entropy(logits, labels)

def alignment_contrastive_loss(local_rep, target_rep, temperature, device="cpu"):
    num = local_rep.shape[0]
    cos_dists = torch.tensor([], device=device)
    for i in range(num):
        tmp = torch.tile(local_rep[i], (num, 1))
        cos = F.cosine_similarity(tmp, target_rep).reshape(1, -1)
        cos_dists = torch.cat((cos_dists, cos), dim=0)
        # print(self.cfg['rank'], cos)

    cos_dists /= temperature

    labels = torch.tensor(range(num), device=device, dtype=torch.long)
    return F.cross_entropy(cos_dists, labels)

def locality_preserving_loss(local_rep, target_rep, locality_preserving_k=5, device="cpu"):
    # for i in range(num):
    #     other = random.choice([n for n in range(num) if n != i])
    #     print(cos_dists[i][i].detach(), cos_dists[i][other].detach())

    # print(self.cfg['rank'])
    # print("pos sum", sum(cos_dists[i][i].detach() for i in range(num)))
    # print("loss sum", loss.detach())

    # norm2 = lambda u, v: ((u-v)**2).sum()
    # k = self.cfg['locality_preserving_k'] + 1
    nbrs = NearestNeighbors(n_neighbors=locality_preserving_k + 1,
                            algorithm='ball_tree')
                            # metric="pyfunc",
                            # metric_params={"func": norm2})
    nbrs = nbrs.fit(target_rep)
    alpha = nbrs.kneighbors_graph(target_rep, mode='distance')
    # g = g.eliminate_zeros()
    alpha.data = np.exp(-alpha.data)
    alphaT = torch.tensor(alpha.toarray(), device=device)

    # dists = scidist.squareform(scidist.cdist(local_rep, norm2))

    dists = torch.cdist(local_rep, local_rep)
    return torch.sum(torch.mul(dists, alphaT))
