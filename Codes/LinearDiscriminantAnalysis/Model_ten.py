import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
class LDA:
    def __init__(self):
        pass
    def train(self, x_train, y_train, k):
        '''
        x_train: training data
        y_train: training labels
        k: target dimension
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        print("Training Data Shape:\t{}".format(x_train.shape))
        print("Training Label Shape:\t{}".format(y_train.shape))
        self.feature_dim = x_train.shape[1]
        self.classes = np.unique(y_train)
        self.x_class_train = {}

        for i in self.classes:
            x_class_i = np.array([x_train[np.where(y_train == i)]]).reshape(-1, self.feature_dim)
            self.x_class_train[i] = x_class_i

        '''
            全局散度矩阵 St = Sw + Sb = sum_i[(x_i-mu)(x_i-mu)^T]
            Sb = St - Sw = sum_i[n_i(mu_i-mu)(mu_i-mu)^T]
            Sw = sum_class[Sw_class] = sum_class[sum_i[(x_i-mu_class)(x_i-mu_class)^T]]
            Sw^-1 Sb W = lambda W
        '''
        self.mu = np.mean(x_train, axis=0)
        self.mu_class = {}
        # compute mu_i
        for i in self.classes:
            self.mu_class[i] = np.mean(self.x_class_train[i], axis=0)
        # compute Sw
        Sw = np.zeros(shape=(self.feature_dim, self.feature_dim))
        for i in self.classes:
            Sw += np.dot((self.x_class_train[i] - self.mu_class[i]).T, self.x_class_train[i] - self.mu_class[i])
        # compute Sb
        Sb = np.zeros(shape=(self.feature_dim, self.feature_dim))
        for i in self.classes:
            Sb_i = np.dot((self.mu_class[i] - self.mu).reshape((self.feature_dim, 1)),
                          (self.mu_class[i] - self.mu).reshape((1, self.feature_dim)))
            Sb += len(self.x_class_train[i]) * Sb_i

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        sorted_indices = np.argsort(eig_vals)
        self.W = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征向量

        self.centers = {}
        for i in self.classes:
            x_transform = np.dot(self.x_class_train[i], self.W)
            self.centers[i] = np.mean(x_transform, axis=0)

    def test(self, x_test, y_test):
        acc = 0.0
        acc_dic = {i:[0, 0] for i in self.classes}
        x_transform = np.dot(x_test, self.W)
        for i in range(x_test.shape[0]):
            dis = 1e10
            max_class = -1
            for class_id in self.classes:
                tmp_dis = np.sum(np.square(x_transform[i]-self.centers[class_id]))
                if tmp_dis < dis:
                    dis = tmp_dis
                    max_class = class_id
            if (max_class == y_test[i]):
                acc += 1
                acc_dic[max_class][0] += 1
            acc_dic[y_test[i]][1] += 1
        return acc/x_test.shape[0], acc_dic

if '__main__' == __name__:
    k_list = [i for i in range(1, 26, 1)]
    res_dic = {}
    x_train = np.load('../../Data/train_x_1.npy')
    y_train = np.load('../../Data/train_y_1.npy')
    x_test = np.load('../../Data/test_x_1.npy')
    y_test = np.load('../../Data/test_y_1.npy')

    '''for k in tqdm(k_list):
        lda = LDA()
        lda.train(x_train, y_train, k)
        acc, acc_dic = lda.test(x_test, y_test)
        print(acc)
        res_dic[k] = [acc, acc_dic]
    np.save("Results/ten/res_dic.npy", res_dic)'''
    model_pca = PCA(n_components=50)
    x_train = model_pca.fit(x_train).transform(x_train)
    lda = LDA()
    lda.train(x_train, y_train, 10)
    x_transform = np.dot(x_train, lda.W)
    model_pca = PCA(n_components=2)
    x_pca_2 = model_pca.fit(x_transform).transform(x_transform)
    N = x_train.shape[0]
    class0 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 0])
    class1 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 1])
    class2 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 2])
    class3 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 3])
    class4 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 4])
    class5 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 5])
    class6 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 6])
    class7 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 7])
    class8 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 8])
    class9 = np.array([x_pca_2[i] for i in range(N) if y_train[i] == 9])
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
    plt.title("True Distribution")
    plt.savefig('true.png')
    plt.show()

