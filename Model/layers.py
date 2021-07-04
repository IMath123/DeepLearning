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

class residual_block(Module):

    def __init__(self, inp, oup, kernel_size, stride, padding, groups=1, bias=True, affine=True):
        super(residual_block, self).__init__()

        self.conv = Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = BatchNorm2d(oup, affine=affine)
        self.relu = ReLU(True)

        if isinstance(padding, int):
            padding = [padding, padding]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.residual = inp == oup and stride == 1 and (2 * padding[0] - kernel_size[0] + 1 == 0) and (2 * padding[1] - kernel_size[1] + 1 == 0)

    def forward(self, input):
        if self.residual:
            return input + self.relu(self.bn(self.conv(input)))
        else:
            return self.relu(self.bn(self.conv(input)))

class add(Module):

    def __init__(self):
        super(add, self).__init__()

    def forward(self, a, b):
        return a + b


class reshape(Module):

    def __init__(self, *shape):
        super(reshape, self).__init__()

        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)
