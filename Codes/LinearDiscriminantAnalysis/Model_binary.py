import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class LDA:
    def __init__(self, x_train, y_train, k, x_test, y_test):
        self.transform_y(y_train, y_test)
        self.W_list = {}
        self.center_list = {}
        self.neg_center_list = {}

        self.x_train = x_train
        self.x_test = x_test
        self.k = k

        self.res_dic = {}
        for i in self.classes:
            W, center, neg_center = self.train(x_train, self.y_train_list[i], i)
            self.W_list[i] = W
            self.center_list[i] = center
            self.neg_center_list[i] = neg_center
            self.res_dic[i] = self.test_class(i, x_test, self.y_test_list[i])
        acc, acc_dic = self.test(x_test, y_test)
        self.res_dic['acc'] = acc
        self.res_dic['acc_dic'] = acc_dic
        print(self.k, acc)

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

    def test_class(self, class_id, x_test, y_test):
        acc = 0.0
        for i in range(x_test.shape[0]):
            x_transform = np.dot(x_test[i], self.W_list[class_id])
            dis_0 = np.mean(np.square(x_transform - self.neg_center_list[class_id]))
            dis_1 = np.mean(np.square(x_transform - self.center_list[class_id]))
            if (dis_0 < dis_1):
                prediction = 0
            else:
                prediction = 1
            acc += (prediction == y_test[i])
        print(class_id, acc/x_test.shape[0])
        return acc

    def test(self, x_test, y_test):
        acc = 0.0
        acc_dic = {i: [0, 0] for i in self.classes}
        for i in range(x_test.shape[0]):
            dis = 1e10
            max_class = -1
            for class_id in self.classes:
                x_transform = np.dot(x_test[i], self.W_list[class_id])
                tmp_dis = np.sum(np.square(x_transform - self.center_list[class_id]))
                if tmp_dis < dis:
                    dis = tmp_dis
                    max_class = class_id
            if (max_class == y_test[i]):
                acc += 1
                acc_dic[max_class][0] += 1
            acc_dic[y_test[i]][1] += 1
        return acc / x_test.shape[0], acc_dic


    def train(self, x_train, y_train, class_id):
        feature_dim = x_train.shape[1]
        #print(np.where(y_train.shape==1))

        x_train_0 = np.array([x_train[np.where(y_train == 0)[0]]]).reshape(-1, feature_dim)
        x_train_1 = np.array([x_train[np.where(y_train == 1)[0]]]).reshape(-1, feature_dim)

        mu_0 = np.mean(x_train_0, axis=0)
        mu_1 = np.mean(x_train_1, axis=0)
        Sb = np.dot((mu_0-mu_1).reshape((feature_dim, 1)), (mu_0-mu_1).reshape((1, feature_dim)))

        sigma_0 = np.dot((x_train_0 - mu_0).T, (x_train_0 - mu_0))
        sigma_1 = np.dot((x_train_1 - mu_1).T, (x_train_1 - mu_1))
        Sw = sigma_0 + sigma_1

        #W = np.dot(np.mat(Sw).I, (mu_0-mu_1).reshape((feature_dim, 1)))

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        sorted_indices = np.argsort(eig_vals)
        W = eig_vecs[:, sorted_indices[:-self.k - 1:-1]]  # 提取前k个特征向量
        x_transform_0 = np.dot(x_train_0, W)
        x_transform_1 = np.dot(x_train_1, W)
        center = np.mean(x_transform_1, axis=0)
        neg_center = np.mean(x_transform_0, axis=0)

        '''plt.scatter(x_transform_0[:, 0], x_transform_0[:, 1], marker='o')
        plt.scatter(x_transform_1[:, 0], x_transform_1[:, 1], marker='+')
        plt.title('Class\t{}'.format(class_id))
        plt.show()'''
        return W, center, neg_center


if '__main__' == __name__:
    k_list = [i for i in range(1, 26, 1)]
    res_dic = {}
    x_train = np.load('../../Data/train_x_1.npy')
    y_train = np.load('../../Data/train_y_1.npy')
    x_test = np.load('../../Data/test_x_1.npy')
    y_test = np.load('../../Data/test_y_1.npy')
    for k in tqdm(k_list):
        lda = LDA(x_train, y_train, k, x_test, y_test)
        res_dic[k] = lda.res_dic
    np.save("Results/binary/res_dic.npy", res_dic)

