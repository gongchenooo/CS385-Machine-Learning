import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
from tqdm import tqdm
class LogisticRegression(object):
    def __init__(self, lr=0.01, reg_lam=0.01, epochs=1000):
        self.lr = lr
        self.reg_lam = reg_lam
        self.epochs = epochs

    def sigmoid(self, y_predict):
        return 1 / (1 + np.exp(-y_predict))

    def bias_x(self, x):
        # add bias one
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    def loss(self, y_predict, y_true, class_id):
        '''
        :param y_predict: (data_num, class_num)
        :param y_true: (data_num, class_num)
        :return:
        '''
        # mean_i[y_i*x_i*beta - log(1+exp(x_i*beta))] + lambda * |beta|
        return -np.mean(y_true * y_predict - np.log(1 + np.exp(y_predict))) + self.reg_lam * np.linalg.norm(self.beta_list[class_id], ord=1)


    def gradient(self, x, y_true, y_predict, class_id):
        # mean_i[(p_i-y_i)*x_i] +
        '''
        :param x: (data num, 325)
        :param y_true: (data_num, class_num)
        :param y_predict: (data_num, class_num)
        :return:
        '''
        x = np.reshape(x, [x.shape[0], x.shape[1], 1])
        y_true = np.reshape(y_true, [y_true.shape[0], 1, y_true.shape[1]])
        y_predict = np.reshape(y_predict, [y_predict.shape[0], 1, y_predict.shape[1]])
        lasso_grad = np.array(self.beta_list[class_id]>0) * 2 - 1
        return np.mean(np.matmul(x, y_predict-y_true), axis=0) + self.reg_lam * lasso_grad

    def accuracy(self, y_predict_list, y_true):
        y_predict = np.argmax(y_predict_list, axis=0).reshape([-1])
        y_true = y_true.reshape([-1])
        acc_class = {i: [0, 0] for i in range(self.num_classes)}
        acc_cnt = 0
        for i in range(y_predict.shape[0]):
            acc_class[y_true[i]][1] += 1  # 真实的类数量+1
            if y_predict[i] == y_true[i]:
                acc_class[y_true[i]][0] += 1  # 正确分类的数量
                acc_cnt += 1
        # print(acc_class)
        return acc_cnt / y_predict.shape[0], acc_class

    def forward(self, x, y_true, class_id):
        y_predict = np.dot(x, self.beta_list[class_id])
        y_predict = self.sigmoid(y_predict)
        loss = self.loss(y_predict, y_true, class_id)
        return y_predict, loss

    def transform_y(self, y_train, y_test):
        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)  # number of classes
        self.y_train = np.reshape(y_train, (y_train.shape[0], 1))
        self.y_test = np.reshape(y_test, (y_test.shape[0], 1))
        self.y_train_list = []
        self.y_test_list = []
        self.label_map = {}  # {label: index}
        # generate label for each class
        for i in range(self.num_classes):
            self.label_map[self.classes[i]] = i  # classes[i]: i

            y_class_i = np.zeros(self.y_train.shape)
            y_class_i[np.where(self.y_train == self.classes[i])] = 1
            self.y_train_list.append(y_class_i)

            y_class_i = np.zeros(self.y_test.shape)
            y_class_i[np.where(self.y_test == self.classes[i])] = 1
            self.y_test_list.append(y_class_i)

    def train(self, x_train, y_train, x_test, y_test):
        x_train = self.bias_x(x_train)
        x_test = self.bias_x(x_test)
        print("x_train:{}\ty_train:{}".format(x_train.shape, y_train.shape))
        print("x_test:{}\ty_test:{}".format(x_test.shape, y_test.shape))

        self.transform_y(y_train, y_test)
        self.beta_list = [0.001 * np.random.randn(x_train.shape[1], 1) for i in range(self.num_classes)]

        res_dic = {"loss_train": [], "accuracy_train": [], "acc_class_train": [],
                   "loss_test": [], "accuracy_test": [], "acc_class_test": [], "beta": []}
        for i in tqdm(range(self.epochs)):
            avg_loss = 0.0
            y_predict_list = []
            for class_id in range(self.num_classes):
                y_predict_train, loss_train = self.forward(x_train, self.y_train_list[class_id], class_id)
                gradient = self.gradient(x_train, self.y_train_list[class_id], y_predict_train, class_id)
                self.beta_list[class_id] -= self.lr * gradient
                avg_loss += loss_train
                y_predict_list.append(y_predict_train)
            accuracy_train, acc_class_train = self.accuracy(y_predict_list, y_train)
            loss_train = avg_loss / self.num_classes

            if (i%10 == 0):
                avg_loss = 0.0
                y_predict_list = []
                for class_id in range(self.num_classes):
                    y_predict_test, loss_test = self.forward(x_test, self.y_test_list[class_id], class_id)
                    avg_loss += loss_test
                    y_predict_list.append(y_predict_test)
                loss_test = avg_loss / self.num_classes
                accuracy_test, acc_class_test = self.accuracy(y_predict_list, y_test)
                print("[{}/{}]\tTrain Loss:{:.4f}\tTrain Accuracy:{:.4f}\tTest Loss:{:.4f}\tTest Accuracy:{:.4f}"
                      .format(i, self.epochs, loss_train, accuracy_train, loss_test, accuracy_test))

                res_dic["loss_train"].append(loss_train)
                res_dic["accuracy_train"].append(accuracy_train)
                res_dic["acc_class_train"].append(acc_class_train)
                res_dic["loss_test"].append(loss_test)
                res_dic["accuracy_test"].append(accuracy_test)
                res_dic["acc_class_test"].append(acc_class_test)
                res_dic["beta"].append(self.beta_list)
                np.save("Results/binary/res_dic_binary_{}.npy".format(self.reg_lam), res_dic)


if __name__ == '__main__':
    x_train = np.load('../../Data/train_x_1.npy')
    y_train = np.load('../../Data/train_y_1.npy')
    x_test = np.load('../../Data/test_x_1.npy')
    y_test = np.load('../../Data/test_y_1.npy')
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


