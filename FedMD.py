import torch
from torch import nn
import numpy as np

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
                 layer1, layer2, layer3, layer4, dropout,
                 projection_size = None,
                 n_classes = 10,
                 input_size = (3, 32, 32)):
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
            tmp1 = calc_size(calc_size(output_size[1], 2, 2, 0), 2, 2, 0)
            tmp2 = calc_size(calc_size(output_size[2], 2, 2, 0), 2, 2, 0)
            output_size = (layer2, tmp1, tmp2)
        else:
            self.layer2 = nn.Identity()

        if layer3:
            self.layer3 = nn.Sequential(
                nn.Conv2d(output_size[0], layer3, kernel_size = 3, stride = 1, padding = 0),
                nn.BatchNorm2d(layer3),
                nn.ReLU(),
                nn.Dropout(dropout),
                #nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
            )
            tmp1 = calc_size(output_size[1], 3, 1, 0)
            tmp2 = calc_size(output_size[2], 3, 1, 0)
            output_size = (layer3, tmp1, tmp2)
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
            tmp1 = calc_size(calc_size(output_size[1], 2, 2, 0) if layer3 else output_size[1], 3, 1, 1)
            tmp2 = calc_size(calc_size(output_size[2], 2, 2, 0) if layer3 else output_size[2], 3, 1, 1)
            output_size = (layer4, tmp1, tmp2)
        else:
            self.layer4 = nn.Identity()

        if not projection_size:
            self.projection_head = nn.Flatten()
        else:
            self.projection_head = nn.Linear(int(np.prod(output_size)), projection_size)
            output_size = projection_size

        self.output = nn.Linear(int(np.prod(output_size)), n_classes, bias = False)

    def forward(self, x, output='logits'):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.flatten(start_dim=1)

        rep = self.projection_head(x)
        if output == 'rep_only':
            return rep

        logits = self.output(rep)
        if output == 'both':
            return logits, rep
        else:
            return logits
