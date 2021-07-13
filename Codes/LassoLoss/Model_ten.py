import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
from tqdm import tqdm
class LogisticRegression(object):

    def __init__(self, lr=0.01, reg_lam=0.1, epochs=10000):
        self.reg_lam = reg_lam
        self.lr = lr
        self.epochs = epochs

    def softmax(self, y_predict):
        y_predict = np.exp(y_predict)
        return y_predict / np.reshape(np.sum(y_predict, axis=1), newshape=[-1, 1])

    def bias_x(self, x):
        # add bias one
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    def loss(self, y_predict, y_true):
        '''
        :param y_predict: (data_num, class_num)
        :param y_true: (data_num, class_num)
        :return:
        '''
        # negative cross entropy: mean_i[y_i*log(p_i)] + lam*|beta|
        labels_true = np.where(y_true == 1)
        return -np.mean(np.log(y_predict[labels_true])) + self.reg_lam * np.linalg.norm(self.beta, ord=1)


    def gradient(self, x, y_true, y_predict):
        # mean_i[(p_i-y_i)*x_i] + lam * beta
        '''
        :param x: (data num, 325)
        :param y_true: (data_num, class_num)
        :param y_predict: (data_num, class_num)
        :return:
        '''
        x = np.reshape(x, [x.shape[0], x.shape[1], 1])
        y_true = np.reshape(y_true, [y_true.shape[0], 1, y_true.shape[1]])
        y_predict = np.reshape(y_predict, [y_predict.shape[0], 1, y_predict.shape[1]])
        lasso_grad = (self.beta>0) * 2 - 1
        return np.mean(np.matmul(x, y_predict-y_true), axis=0) + self.reg_lam * lasso_grad

    def accuracy(self, y_predict, y_true):
        y_predict = np.argmax(y_predict, axis=1)
        y_true = np.argmax(y_true, axis=1)
        acc_class = {i: [0, 0] for i in range(self.num_classes)}
        acc_cnt = 0
        for i in range(y_predict.shape[0]):
            acc_class[y_true[i]][1] += 1  # 真实的类数量+1
            if y_predict[i] == y_true[i]:
                acc_class[y_true[i]][0] += 1  # 正确分类的数量
                acc_cnt += 1
        # print(acc_class)
        return acc_cnt / y_predict.shape[0], acc_class


    def forward(self, x, y_true):
        y_predict = np.dot(x, self.beta)
        y_predict = self.softmax(y_predict)
        loss = self.loss(y_predict, y_true)
        accuracy, acc_class = self.accuracy(y_predict, y_true)
        return y_predict, loss, accuracy, acc_class

    def train(self, x_train, y_train, x_test, y_test):
        x_train = self.bias_x(x_train)
        x_test = self.bias_x(x_test)
        print("x_train:{}\ty_train:{}".format(x_train.shape, y_train.shape))
        print("x_test:{}\ty_test:{}".format(x_test.shape, y_test.shape))

        self.num_classes = y_train.shape[1]
        self.beta = np.zeros((x_train.shape[1], self.num_classes))

        res_dic = {"loss_train": [], "accuracy_train": [], "acc_class_train": [],
                   "loss_test": [], "accuracy_test": [], "acc_class_test": [], "beta": []}
        for i in tqdm(range(self.epochs)):
            y_predict_train, loss_train, accuracy_train, acc_class_train = self.forward(x_train, y_train)
            gradient = self.gradient(x_train, y_train, y_predict_train)
            self.beta -= self.lr * gradient
            if (i%10 == 0):
                y_predict_test, loss_test, accuracy_test, acc_class_test = self.forward(x_test, y_test)
                print("[{}/{}]\tTrain Loss:{:.4f}\tTrain Accuracy:{:.4f}\tTest Loss:{:.4f}\tTest Accuracy:{:.4f}"
                      .format(i, self.epochs, loss_train, accuracy_train, loss_test, accuracy_test))
                res_dic["loss_train"].append(loss_train)
                res_dic["accuracy_train"].append(accuracy_train)
                res_dic["acc_class_train"].append(acc_class_train)
                res_dic["loss_test"].append(loss_test)
                res_dic["accuracy_test"].append(accuracy_test)
                res_dic["acc_class_test"].append(acc_class_test)
                res_dic["beta"].append(self.beta)
                np.save("Results/ten/res_dic_{}.npy".format(self.reg_lam), res_dic)



def transform_label(labels):
    new_labels = np.zeros(shape=[labels.shape[0], 10])
    for i in range(labels.shape[0]):
        new_labels[i][labels[i]] = 1
    return new_labels

if __name__ == '__main__':
    x_train = np.load('../../Data/train_x_1.npy')
    y_train = transform_label(np.load('../../Data/train_y_1.npy'))
    x_test = np.load('../../Data/test_x_1.npy')
    y_test = transform_label(np.load('../../Data/test_y_1.npy'))

    np.random.seed(123)
    train_num = x_train.shape[0]
    test_num = x_test.shape[0]
    train_choice = np.random.choice(range(train_num), train_num // 7, replace=False)
    test_choice = np.random.choice(range(test_num), test_num // 7, replace=False)
    print(train_choice)
    x_train = x_train[train_choice]
    y_train = y_train[train_choice]
    x_test = x_test[test_choice]
    y_test = y_test[test_choice]

    lam_list = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    for lam in lam_list:
        logistic = LogisticRegression(epochs=3000, reg_lam=lam, lr=0.1)
        logistic.train(x_train, y_train, x_test, y_test)