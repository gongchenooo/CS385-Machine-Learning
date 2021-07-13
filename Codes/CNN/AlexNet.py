import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from collections import *

class AlexNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)),
            ('r1', nn.ReLU(inplace=True)),
            ('m1', nn.MaxPool2d(kernel_size=2)),
            ('c2', nn.Conv2d(64, 192, kernel_size=3, padding=1)),
            ('r2', nn.ReLU(inplace=True)),
            ('m2', nn.MaxPool2d(kernel_size=2)),
            ('c3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('r3', nn.ReLU(inplace=True)),
            ('c4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('r4', nn.ReLU(inplace=True)),
            ('c5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('r5', nn.ReLU(inplace=True)),
            ('m5', nn.MaxPool2d(kernel_size=2))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('d6', nn.Dropout()),
            ('l6', nn.Linear(256 * 2 * 2, 4096)),
            ('r6', nn.ReLU(inplace=True)),
            ('d7', nn.Dropout()),
            ('l7', nn.Linear(4096, 4096)),
            ('r7', nn.ReLU(inplace=True)),
            ('l8', nn.Linear(4096, num_classes))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x



