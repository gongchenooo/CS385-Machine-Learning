import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_dim, g_feature_num):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(z_dim, g_feature_num*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(g_feature_num*8)
        self.tconv2 = nn.ConvTranspose2d(g_feature_num*8, g_feature_num*4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(g_feature_num*4)
        self.tconv3 = nn.ConvTranspose2d(g_feature_num*4, g_feature_num*2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(g_feature_num*2)
        self.tconv4 = nn.ConvTranspose2d(g_feature_num*2, g_feature_num, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(g_feature_num)
        self.tconv5 = nn.ConvTranspose2d(g_feature_num, 3, 4, 2, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = F.tanh(self.tconv5(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, d_feature_num):
        super().__init__()
        self.conv1 = nn.Conv2d(3, d_feature_num, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(d_feature_num, d_feature_num*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(d_feature_num*2)
        self.conv3 = nn.Conv2d(d_feature_num*2, d_feature_num*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(d_feature_num*4)
        self.conv4 = nn.Conv2d(d_feature_num*4, d_feature_num*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(d_feature_num*8)
        self.conv5 = nn.Conv2d(d_feature_num*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = F.sigmoid(self.conv5(x))

        return x