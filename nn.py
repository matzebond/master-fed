import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
import copy
from typing import List, Callable, Optional


def avg_params(models: List[nn.Module]) -> nn.Module:
    assert len(models) > 0
    model_global = copy.deepcopy(models[0])

    for model in models:
        for (global_p,model_p) in zip(model_global.parameters(), model.parameters()):
            with torch.no_grad():
                global_p += model_p

    for p in model_global.parameters():
        with torch.no_grad():
            p /= len(models)

    for model in models:
        for (global_p,model_p) in zip(model_global.parameters(), model.parameters()):
            with torch.no_grad():
                model_p.copy_(global_p)

    return model_global


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


class KLDivSoftmaxLoss(nn.modules.loss._WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(KLDivSoftmaxLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(F.softmax(input), target,
                        reduction=self.reduction, log_target=self.log_target)
