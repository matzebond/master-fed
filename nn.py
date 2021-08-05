import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
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

