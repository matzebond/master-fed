import torch
from typing import List

def train(dataloader, model, loss_fn, optimizer, count_correct=True, verbose=True):
    model.train()
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Compute training statistics
        train_loss += loss.item()
        if count_correct:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 and verbose:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= size
    correct /= size
    return train_loss, correct


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct


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
        

# from torch.nn import _WeightedLoss

# class _Loss(Module):
#     reduction: str

#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_Loss, self).__init__()
#         if size_average is not None or reduce is not None:
#             self.reduction = _Reduction.legacy_get_string(size_average, reduce)
#         else:
#             self.reduction = reduction


# class _WeightedLoss(_Loss):
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
#         self.register_buffer('weight', weight)



# class CrossEntropyLoss(_WeightedLoss):
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
#                  reduce=None, reduction: str = 'mean') -> None:
#         super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         assert self.weight is None or isinstance(self.weight, Tensor)
#         return F.cross_entropy(input, target, weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction)
