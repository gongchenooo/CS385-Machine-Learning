import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
from tqdm import tqdm
class LogisticRegression(object):

    def __init__(self, lr=0.01, epochs=10000):
        self.num_classes = 1
        self.lr = lr
        self.epochs = epochs

    def kernel(self, x_i_list, x_j_list):
        


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
        # negative cross entropy: mean_i[y_i*log(p_i)]
        labels_true = np.where(y_true == 1)
        return -np.mean(np.log(y_predict[labels_true]))


    def gradient(self, x, y_true, y_predict):
        # mean_i[(p_i-y_i)*x_i]
        '''
        :param x: (data num, 325)
        :param y_true: (data_num, class_num)
        :param y_predict: (data_num, class_num)
        :return:
        '''
        x = np.reshape(x, [x.shape[0], x.shape[1], 1])
        y_true = np.reshape(y_true, [y_true.shape[0], 1, y_true.shape[1]])
        y_predict = np.reshape(y_predict, [y_predict.shape[0], 1, y_predict.shape[1]])
        return np.mean(np.matmul(x, y_predict-y_true), axis=0)

    def accuracy(self, y_predict, y_true):
        compare = np.argmax(y_predict, axis=1) == np.argmax(y_true, axis=1)
        return np.mean(compare)


    def forward(self, x, y_true):
        y_predict = np.dot(x, self.beta)
        y_predict = self.softmax(y_predict)
        loss = self.loss(y_predict, y_true)
        accuracy = self.accuracy(y_predict, y_true)
        return y_predict, loss, accuracy

    def train(self, x_train, y_train, x_test, y_test):
        x_train = self.bias_x(x_train)
        x_test = self.bias_x(x_test)
        print("x_train:{}\ty_train:{}".format(x_train.shape, y_train.shape))
        print("x_test:{}\ty_test:{}".format(x_test.shape, y_test.shape))

        self.num_classes = y_train.shape[1]
        self.beta = np.zeros((x_train.shape[1], self.num_classes))

        res_dic = {"loss_train":[], "accuracy_train":[], "loss_test":[], "accuracy_test":[]}
        for i in tqdm(range(self.epochs)):
            y_predict_train, loss_train, accuracy_train = self.forward(x_train, y_train)
            gradient = self.gradient(x_train, y_train, y_predict_train)
            self.beta -= self.lr * gradient
            if (i%10 == 0):
                y_predict_test, loss_test, accuracy_test = self.forward(x_test, y_test)
                print("[{}/{}]\tTrain Loss:{:.4f}\tTrain Accuracy:{:.4f}\tTest Loss:{:.4f}\tTest Accuracy:{:.4f}"
                      .format(i, self.epochs, loss_train, accuracy_train, loss_test, accuracy_test))
                res_dic["loss_train"].append(loss_train)
                res_dic["accuracy_train"].append(accuracy_train)
                res_dic["loss_test"].append(loss_test)
                res_dic["accuracy_test"].append(accuracy_test)
                np.save("res_dic.npy", res_dic)


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

    logistic = LogisticRegression(epochs=10000, lr=0.1)
    logistic.train(x_train, y_train, x_test, y_test)


