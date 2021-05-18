import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)

from typing import List, Callable, Optional

def avg_params(models: List[nn.Module]) -> None:
    params = []
    for p in models[0].parameters():
        d = torch.zeros_like(p)
        params.append(d)

    for model in models:
        for (p,model_p) in zip(params, model.parameters()):
            p += model_p

    for p in params:
        p /= len(models)

    for model in models:
        for n,(p,model_p) in enumerate(zip(params, model.parameters())):
            with torch.no_grad():
                model_p.copy_(p)


def reset_all_parameters(model):
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()
    else:
        pass


class KLDivSoftmaxLoss(nn.modules.loss._WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(KLDivSoftmaxLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(F.softmax(input), target,
                        reduction=self.reduction, log_target=self.log_target)


# def kl_div_softmax(
#     input: Tensor,
#     target: Tensor,
#     size_average: Optional[bool] = None,
#     reduce: Optional[bool] = None,
#     reduction: str = "mean",
#     log_target: bool = False,
# ) -> Tensor:
#     if has_torch_function_variadic(input, target):
#         input = F.softmax(input)
#         return handle_torch_function(
#             kl_div,
#             (input, target),
#             input,
#             target,
#             size_average=size_average,
#             reduce=reduce,
#             reduction=reduction,
#             log_target=log_target,
#         )
#     if size_average is not None or reduce is not None:
#         reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
#     else:
#         if reduction == "mean":
#             warnings.warn(
#                 "reduction: 'mean' divides the total loss by both the batch size and the support size."
#                 "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
#                 "'mean' will be changed to behave the same as 'batchmean' in the next major release."
#             )

#         # special case for batchmean
#         if reduction == "batchmean":
#             reduction_enum = _Reduction.get_enum("sum")
#         else:
#             reduction_enum = _Reduction.get_enum(reduction)

#     reduced = torch.kl_div(F.softmax(input), target, reduction_enum, log_target=log_target)

#     if reduction == "batchmean" and input.dim() != 0:
#         reduced = reduced / input.size()[0]

#     return reduced
