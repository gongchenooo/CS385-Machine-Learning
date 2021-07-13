import torch
import torchvision
import torchvision.transforms as tf
from skimage import feature as ft
import numpy as np
from skimage import io

train_data = torchvision.datasets.SVHN(root='Data/', split="train", download=True, transform=tf.ToTensor())
test_data = torchvision.datasets.SVHN(root='Data/', split="test", download=True, transform=tf.ToTensor())

train_X = []
train_Y = []
test_X = []
test_Y = []

for i in range(len(train_data)):
    train_X.append(np.array(train_data[i][0]))
    train_Y.append(train_data[i][1])
for i in range(len(test_data)):
    test_X.append(np.array(test_data[i][0]))
    test_Y.append(test_data[i][1])
print(len(train_X))
print(len(test_Y))
np.save("Data/train_x_0.npy", train_X)
np.save("Data/train_y_0.npy", train_Y)
np.save("Data/test_x_0.npy", test_X)
np.save("Data/test_y_0.npy", test_Y)


train_X = np.load("Data/train_x_0.npy")
train_Y = np.load("Data/train_y_0.npy")
test_X = np.load("Data/test_x_0.npy")
test_Y = np.load("Data/test_y_0.npy")
train_X = [ft.hog(i.transpose((2, 1, 0))) for i in train_X]
test_X = [ft.hog(i.transpose((2, 1, 0))) for i in test_X]
print(len(train_X))
print(len(test_X))
np.save("train_x_1.npy", train_X)
np.save("train_y_1.npy", train_Y)
np.save("test_x_1.npy", test_X)
np.save("test_y_1.npy", test_Y)