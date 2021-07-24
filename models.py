import torch
from torch import nn
import numpy as np
from typing import Union, List, Optional, Tuple

calc_size = lambda x, k, s, p: (x + 2*p - k)/s + 1

# used in most models
def build_projection_block(
        definition: Union[None, int, List[int]] = None,
        last_nonlinear: bool = True,
        input_size: Tuple[int, int, int] = (3, 32, 32),
) -> Tuple[nn.Module, int]:
    output_size = input_size
    if definition is None:
        projection = nn.Flatten()
        output_size = int(np.prod(output_size))
    else:
        output_size = int(np.prod(output_size))
        proj = [nn.Flatten()]
        if isinstance(definition, int):
            definition = [definition]
        for i in definition:
            if i == 0:
                proj.append(nn.Linear(output_size, output_size))
            else:
                proj.append(nn.Linear(output_size, i))
            output_size = proj[-1].out_features
            proj.append(nn.ReLU())
        if last_nonlinear:
            projection = nn.Sequential(*proj)
        else:
            projection = nn.Sequential(*proj[:-1])
        output_size = proj[-2].out_features

    return projection, output_size

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

        if layer1:
            self.layer1 = nn.Sequential(
                nn.Conv2d(input_size[0], layer1, kernel_size = 3, stride = 1, padding = 1), # padding = "valid"
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


        self.projection, output_size = build_projection_block(
            projection, projection_nonlinear, output_size)

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


# from the paper Learning Student Networks via Feature Embedding - 2021
class LPP(nn.Module):
    hyper = [[([96], 4), ([96], 4), ([96], 4)], # Teacher
             [([16, 16, 16], 2), ([32, 32, 32], 2), ([48, 48, 64], 8)], # Student 1
             [([16, 32, 32], 2), ([48, 64, 80], 2), ([96, 96, 128], 8)], # Student 2
             [([32, 48, 64, 64], 2), ([80, 80, 80, 80], 2), ([128, 128, 128], 8)], # Student 3
             [([32, 32, 32, 48, 48], 2), ([80, 80, 80, 80, 80, 80], 2), ([128, 128, 128, 128, 128, 128], 8)], # Student 4
             ]

    def __init__(self,
                 *conv_def,
                 dropout = 0.2,
                 projection: Union[None, int, List[int]] = 500,
                 projection_nonlinear: bool = False,
                 n_classes: int = 10,
                 input_size: Tuple[int, int, int] = (3, 32, 32)):
        super(LPP, self).__init__()
        output_size = input_size

        conv_block = []
        for group in conv_def:
            layers = []
            for feat in group[0]:
                layers.append(nn.Conv2d(output_size[0], feat,
                                        kernel_size=3, stride=1, padding=1))
                w = calc_size(output_size[1], 3, 1, 1)
                h = calc_size(output_size[2], 3, 1, 1)
                output_size = (feat, w, h)

            layers.append(nn.BatchNorm2d(output_size[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            pool = group[1]
            layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
            w = calc_size(output_size[1], pool, pool, 0)
            h = calc_size(output_size[2], pool, pool, 0)
            output_size = (output_size[0], w, h)
            conv_block.append(nn.Sequential(*layers))
            print(output_size)
        self.conv = nn.Sequential(*conv_block)

        self.projection, output_size = build_projection_block(
            projection, projection_nonlinear, output_size)

        self.output = nn.Linear(output_size, n_classes, bias=False)

    def forward(self, x, output='logits', from_rep=None):
        if from_rep is None:
            x = self.conv(x)

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


class LeNet_plus_plus(nn.Module):
    """
    Defines the network architecture for LeNet++.
    Use the options for different approaches:
    background_class: Classification with additional class for negative classes
    ring_approach: ObjectoSphere Loss applied if True
    knownsMinimumMag: Minimum Magnitude allowed for samples belonging to one of the Known Classes if ring_approach is True
    """
    hyper = [[32, 32, 64, 64, 128, 128, 3],
             [16, 16, 32, 32, 64, 64, 2]]

    def __init__(self,
                 *defi,
                 projection: int = 3,
                 input_size: tuple[int, int, int] = (1, 28, 28),
                 n_classes: int = 10):
        super(LeNet_plus_plus, self).__init__()
        output_size = input_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], defi[0], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(defi[0], defi[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(defi[1]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        output_size = (defi[1], 14, 14)  # 28 X 28 --> 14 X 14

        self.layer2 = nn.Sequential(
            nn.Conv2d(defi[1], defi[2], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(defi[2], defi[3], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(defi[3]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        output_size = (defi[3], 7, 7)  # 14 X 14 --> 7 X 7

        self.layer3 = nn.Sequential(
            nn.Conv2d(defi[3], defi[4], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(defi[4], defi[5], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(defi[5]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        output_size = (defi[5], 3, 3)  # 7 X 7 --> 3 X 3

        projection = defi[6]
        self.projection, output_size = build_projection_block(
            projection, True, output_size)

        self.output = nn.Linear(output_size, n_classes)

    def forward(self, x, output='logits', from_rep=None):
        if from_rep is None:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            # x = self.layer4(x)

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
