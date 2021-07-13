import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
from tqdm import tqdm
from kernel import *
import gc
class LogisticRegression(object):
    def __init__(self, lr=0.01, epochs=10000, kernel_name='rbf'):
        self.lr = lr
        self.epochs = epochs
        self.kernel_name = kernel_name
    def kernel(self, x_i_list, x_j_list, choose='train'):
        try:
            kernel_matrix = np.load(self.kernel_name+'_'+choose+'.npy')
            print('Loading Kernel Matrix')
            return kernel_matrix
        except:
            print('Computing Kernel Matrix')
            pass
        if self.kernel_name == 'rbf':
            # 归一化试一下作用
            kernel_matrix = rbf(x_i_list, x_j_list, sigma=1)
            kernel_matrix = (kernel_matrix - kernel_matrix.mean()) / kernel_matrix.std()
            np.save('rbf_{}.npy'.format(choose), kernel_matrix)
            print("Already Computing")
        elif self.kernel_name == 'poly':
            kernel_matrix = poly(x_i_list, x_j_list, degree=2)
            kernel_matrix = (kernel_matrix - kernel_matrix.mean()) / kernel_matrix.std()
            np.save('poly_{}.npy'.format(choose), kernel_matrix)
        elif self.kernel_name == 'cos':
            # 归一化试一下作用
            kernel_matrix = cos(x_i_list, x_j_list)
            kernel_matrix = (kernel_matrix - kernel_matrix.mean()) / kernel_matrix.std()
            np.save('cos_{}.npy'.format(choose), kernel_matrix)
        elif self.kernel_name == 'sigmoid':
            # 归一化试一下作用
            kernel_matrix = sigmoid(x_i_list, x_j_list, alpha=1e-4, c=1e-2)
            kernel_matrix = (kernel_matrix - kernel_matrix.mean()) / kernel_matrix.std()
            np.save('sigmoid_{}.npy'.format(choose), kernel_matrix)
        else:
            exit('Error: no such kernel!')
        gc.collect()
        return kernel_matrix

    def sigmoid(self, y_predict):
        return 1 / (1 + np.exp(-y_predict))

    def bias_x(self, x):
        # add bias one
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    def loss(self, y_predict, y_true):
        '''
        :param y_predict: (data_num, class_num)
        :param y_true: (data_num, class_num)
        :return:
        '''
        # mean_i[y_i*x_i*beta - log(1+exp(x_i*beta))]
        return -np.mean(y_true * y_predict - np.log(1 + np.exp(y_predict)))


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

    def accuracy(self, y_predict_list, y_true):
        y_predict = np.argmax(y_predict_list, axis=0).reshape([-1])
        y_true = y_true.reshape([-1])
        compare = (y_predict == y_true)
        return np.mean(compare)

    def forward(self, x, y_true, class_id):
        y_predict = np.dot(x, self.beta_list[class_id])
        y_predict = self.sigmoid(y_predict)
        loss = self.loss(y_predict, y_true)
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
        # 试一下bias的效果
        x_train = self.bias_x(x_train)
        x_test = self.bias_x(x_test)
        x_train = self.kernel(x_train, x_train, choose='train')
        x_test = self.kernel(x_test, x_train,  choose='test')
        print("x_train:{}\ty_train:{}".format(x_train.shape, y_train.shape))
        print("x_test:{}\ty_test:{}".format(x_test.shape, y_test.shape))

        self.transform_y(y_train, y_test)
        self.beta_list = [0.001 * np.random.randn(x_train.shape[1], 1) for i in range(self.num_classes)]

        res_dic = {"loss_train":[], "accuracy_train":[], "loss_test":[], "accuracy_test":[]}
        for i in tqdm(range(self.epochs)):
            avg_loss = 0.0
            y_predict_list = []
            for class_id in range(self.num_classes):
                y_predict_train, loss_train= self.forward(x_train, self.y_train_list[class_id], class_id)
                gradient = self.gradient(x_train, self.y_train_list[class_id], y_predict_train)
                self.beta_list[class_id] -= self.lr * gradient
                avg_loss += loss_train
                y_predict_list.append(y_predict_train)
            accuracy_train = self.accuracy(y_predict_list, y_train)
            loss_train = avg_loss / self.num_classes

            if (i%10 == 0):
                avg_loss = 0.0
                y_predict_list = []
                for class_id in range(self.num_classes):
                    y_predict_test, loss_test = self.forward(x_test, self.y_test_list[class_id], class_id)
                    avg_loss += loss_test
                    y_predict_list.append(y_predict_test)
                loss_test = avg_loss / self.num_classes
                accuracy_test = self.accuracy(y_predict_list, y_test)
                print("[{}/{}]\tTrain Loss:{:.4f}\tTrain Accuracy:{:.4f}\tTest Loss:{:.4f}\tTest Accuracy:{:.4f}"
                      .format(i, self.epochs, loss_train, accuracy_train, loss_test, accuracy_test))

                res_dic["loss_train"].append(loss_train)
                res_dic["accuracy_train"].append(accuracy_train)
                res_dic["loss_test"].append(loss_test)
                res_dic["accuracy_test"].append(accuracy_test)
                np.save("res_dic_binary.npy", res_dic)



if __name__ == '__main__':
    x_train = np.load('../../Data/train_x_1.npy')
    y_train = np.load('../../Data/train_y_1.npy')
    x_test = np.load('../../Data/test_x_1.npy')
    y_test = np.load('../../Data/test_y_1.npy')
    np.random.seed(123)
    train_num = x_train.shape[0]
    test_num = x_test.shape[0]
    train_choice = np.random.choice(range(train_num), train_num // 30, replace=False)
    test_choice = np.random.choice(range(test_num), test_num // 30, replace=False)
    print(train_choice)
    x_train = x_train[train_choice]
    y_train = y_train[train_choice]
    x_test = x_test[test_choice]
    y_test = y_test[test_choice]

    logistic = LogisticRegression(epochs=3000, lr=0.1)
    logistic.train(x_train, y_train, x_test, y_test)


