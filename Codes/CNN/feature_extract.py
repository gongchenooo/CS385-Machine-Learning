import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

data_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.SVHN(root='../../Data', split='train', transform=data_tf, download=True)
test_dataset = datasets.SVHN(root='../../Data', split='test', transform=data_tf, download=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
in_channels = 3
num_classes = 10

use_cuda = 0
device = torch.device("cuda:0" if use_cuda else "cpu")
model = torch.load('AlexNet.pt')

x_train = []
y_train = []
x_test = []
y_test = []
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    output = model.features(data)