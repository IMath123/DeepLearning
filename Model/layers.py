import torch
from torch.nn import *
import torch.nn as nn

class conv_bn_relu(Module):

    def __init__(self, inp, oup, kernel_size, stride, padding, groups=1, bias=True, affine=True):
        super(conv_bn_relu, self).__init__()

        self.conv = Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = BatchNorm2d(oup, affine=affine)
        self.relu = ReLU(True)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))

class add(Module):

    def __init__(self):
        super(add, self).__init__()

    def forward(self, a, b):
        return a + b
