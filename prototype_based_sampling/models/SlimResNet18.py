"""
This is the slimmed ResNet as used by Lopez et al. in the GEM paper.

The following code is a modified version of the original code from the Avalanche library.
The original code can be found at and is licensed under the MIT License:
https://github.com/ContinualAI/avalanche/blob/master/avalanche/models/slim_resnet18.py

"""

import torch.nn as nn
from torch.nn.functional import avg_pool2d, relu


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_dim):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_dim = input_dim

        self.conv1 = conv3x3(input_dim[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(self.extract_last_layer(x))
        return out

    def extract_last_layer(self, x):
        bsz = x.size(0)

        out = relu(self.bn1(self.conv1(x.view(bsz, *self.input_dim))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def SlimResNet18(nclasses, input_dim=(1, 28, 28)):
    """Slimmed ResNet18."""
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf=64, input_dim=input_dim)


def ResNet50(nclasses, input_dim=(1, 28, 28)):
    """Slimmed ResNet18."""
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf=64, input_dim=input_dim)


def ResNet101(nclasses, input_dim=(1, 28, 28)):
    return ResNet(BasicBlock, [3, 4, 23, 3], nclasses, nf=64, input_dim=input_dim)


def ResNet152(nclasses, input_dim=(1, 28, 28)):
    return ResNet(BasicBlock, [3, 8, 36, 3], nclasses, nf=64, input_dim=input_dim)
