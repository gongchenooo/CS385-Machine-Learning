# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import matplotlib.pyplot as plt
from gmm import *
import numpy as np
from sklearn.decomposition import PCA
# 设置调试模式
DEBUG = True

pathx = '../../Data/train_x_1.npy'[:10000]
pathy = '../../Data/train_y_1.npy'[:10000]
#pathx = '../CNN/x_train_Alex.npy'
#pathy = '../CNN/y_train_Alex.npy'
# 载入数据
Y = np.load(pathx)
Y_train = np.load(pathy)
model_pca = PCA(n_components=50)
x_pca = model_pca.fit(Y).transform(Y)
Y = x_pca
matY = np.matrix(Y, copy=True)
label = np.load(pathy)
# 模型个数，即聚类的类别个数
K = 10

# 计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, 20)

# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]
# 求当前模型参数下，各模型对样本的响应度矩阵
gamma = getExpectation(matY, mu, cov, alpha)
# 对每个样本，求响应度最大的模型下标，作为其类别标识
category = gamma.argmax(axis=1).flatten().tolist()[0]
print("Successfully!")
# 将每个样本放入对应类别的列表中
Y = np.load(pathx)
model_pca = PCA(n_components=2)
x_pca = model_pca.fit(Y).transform(Y)
Y = x_pca
print(Y.shape)
class0 = np.array([Y[i] for i in range(N) if category[i] == 0])
class1 = np.array([Y[i] for i in range(N) if category[i] == 1])
class2 = np.array([Y[i] for i in range(N) if category[i] == 2])
class3 = np.array([Y[i] for i in range(N) if category[i] == 3])
class4 = np.array([Y[i] for i in range(N) if category[i] == 4])
class5 = np.array([Y[i] for i in range(N) if category[i] == 5])
class6 = np.array([Y[i] for i in range(N) if category[i] == 6])
class7 = np.array([Y[i] for i in range(N) if category[i] == 7])
class8 = np.array([Y[i] for i in range(N) if category[i] == 8])
class9 = np.array([Y[i] for i in range(N) if category[i] == 9])
# 绘制聚类结果
plt.scatter(class0[:, 0], class0[:, 1], label="class0")
plt.scatter(class1[:, 0], class1[:, 1], label="class1")
plt.scatter(class2[:, 0], class2[:, 1], label="class2")
plt.scatter(class3[:, 0], class3[:, 1], label="class3")
plt.scatter(class4[:, 0], class4[:, 1], label="class4")
plt.scatter(class5[:, 0], class5[:, 1], label="class5")
plt.scatter(class6[:, 0], class6[:, 1], label="class6")
plt.scatter(class7[:, 0], class7[:, 1], label="class7")
plt.scatter(class8[:, 0], class8[:, 1], label="class8")
plt.scatter(class9[:, 0], class9[:, 1], label="class9")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.savefig('gmm.png')
plt.show()

class0 = np.array([Y[i] for i in range(N) if Y_train[i] == 0])
class1 = np.array([Y[i] for i in range(N) if Y_train[i] == 1])
class2 = np.array([Y[i] for i in range(N) if Y_train[i] == 2])
class3 = np.array([Y[i] for i in range(N) if Y_train[i] == 3])
class4 = np.array([Y[i] for i in range(N) if Y_train[i] == 4])
class5 = np.array([Y[i] for i in range(N) if Y_train[i] == 5])
class6 = np.array([Y[i] for i in range(N) if Y_train[i] == 6])
class7 = np.array([Y[i] for i in range(N) if Y_train[i] == 7])
class8 = np.array([Y[i] for i in range(N) if Y_train[i] == 8])
class9 = np.array([Y[i] for i in range(N) if Y_train[i] == 9])
plt.scatter(class0[:, 0], class0[:, 1], label="class0")
plt.scatter(class1[:, 0], class1[:, 1], label="class1")
plt.scatter(class2[:, 0], class2[:, 1], label="class2")
plt.scatter(class3[:, 0], class3[:, 1], label="class3")
plt.scatter(class4[:, 0], class4[:, 1], label="class4")
plt.scatter(class5[:, 0], class5[:, 1], label="class5")
plt.scatter(class6[:, 0], class6[:, 1], label="class6")
plt.scatter(class7[:, 0], class7[:, 1], label="class7")
plt.scatter(class8[:, 0], class8[:, 1], label="class8")
plt.scatter(class9[:, 0], class9[:, 1], label="class9")
plt.legend(loc="best")
plt.title("Clustering of GroundTruth")
plt.savefig('original.png')
plt.show()

