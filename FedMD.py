import torch
from torch import nn
import numpy as np
from typing import Union, List, Optional, Tuple

# from the FedMD paper

# basic structure from https://github.com/Tzq2doc/FedMD/blob/master/Neural_Networks.py
# but stride and padding in layer 3 and 4 modified
class FedMD_CIFAR(nn.Module):
    hyper = [[128, 256, None, None, 0.2],
             [128, 128, 192,  None, 0.2],
             [64,  64,  64,   None, 0.2],
             [128, 64,  64,   None, 0.3],
             [64,  64,  128,  None, 0.4],
             [64,  128, 256,  None, 0.2],
             [64,  128, 192,  None, 0.2],
             [128, 192, 256,  None, 0.2],
             [128, 128, 128,  None, 0.3],
             [64,  64,  64,   64,   0.2]]

    def __init__(self,
                 layer1: Optional[int], layer2: Optional[int],
                 layer3: Optional[int], layer4: Optional[int], dropout: float,
                 projection: Union[None, int, List[int]] = None,
                 projection_nonlinear: bool = False,
                 n_classes: int = 10,
                 input_size: Tuple[int, int, int] = (3, 32, 32)):
        super(FedMD_CIFAR, self).__init__()
        output_size = input_size
        calc_size = lambda x, k, s, p: (x + 2*p - k)/s + 1

        if layer1:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, layer1, kernel_size = 3, stride = 1, padding = 1), # padding = "valid"
                nn.BatchNorm2d(layer1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.ZeroPad2d((1,0,1,0)),  # padding = "valid"
                nn.AvgPool2d(kernel_size = 2, stride = 1, padding = 0)
            )
            output_size = (layer1, input_size[1], input_size[2])
        else:
            self.layer1 = nn.Identity()
            output_size = input_size

        if layer2:
            self.layer2 = nn.Sequential(
                nn.Conv2d(output_size[0], layer2, kernel_size = 2, stride = 2, padding = 0),
                nn.BatchNorm2d(layer2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
            )
            w = calc_size(calc_size(output_size[1], 2, 2, 0), 2, 2, 0)
            h = calc_size(calc_size(output_size[2], 2, 2, 0), 2, 2, 0)
            output_size = (layer2, w, h)
        else:
            self.layer2 = nn.Identity()

        if layer3:
            self.layer3 = nn.Sequential(
                nn.Conv2d(output_size[0], layer3, kernel_size = 3, stride = 1, padding = 0),
                nn.BatchNorm2d(layer3),
                nn.ReLU(),
                nn.Dropout(dropout),
                # nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
            )
            w = calc_size(output_size[1], 3, 1, 0)
            h = calc_size(output_size[2], 3, 1, 0)
            output_size = (layer3, w, h)
        else:
            self.layer3 = nn.Identity()

        if layer4:
            self.layer4 = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0) if layer3 else nn.Identity(),
                nn.Conv2d(output_size[0], layer4, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(layer4),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            w = calc_size(calc_size(output_size[1], 2, 2, 0) if layer3 else output_size[1], 3, 1, 1)
            h = calc_size(calc_size(output_size[2], 2, 2, 0) if layer3 else output_size[2], 3, 1, 1)
            output_size = (layer4, w, h)
        else:
            self.layer4 = nn.Identity()

        if projection is None:
            self.projection = nn.Flatten()
            output_size = int(np.prod(output_size))
        else:
            output_size = int(np.prod(output_size))
            proj = [nn.Flatten()]
            if isinstance(projection, int):
                projection = [projection]
            for i in projection:
                if i == 0:
                    proj.append(nn.Linear(output_size, output_size))
                else:
                    proj.append(nn.Linear(output_size, i))
                output_size = proj[-1].out_features
                proj.append(nn.ReLU())
            if projection_nonlinear:
                self.projection = nn.Sequential(*proj)
            else:
                self.projection = nn.Sequential(*proj[:-1])
            output_size = proj[-2].out_features

        self.output = nn.Linear(output_size, n_classes, bias = False)

    def change_classes(self, n_classes):
        in_features = self.output.in_features
        self.output = nn.Linear(in_features, n_classes, bias = False)

    def forward(self, x, output='logits', from_rep=None):
        if from_rep is None:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            rep = self.projection(x)
            if output == 'rep_only':
                return rep
        else:
            rep = from_rep

        logits = self.output(rep)
        if output == 'both':
            return logits, rep
        else:
            return logits
