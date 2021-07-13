import numpy as np
from sklearn import svm
from tqdm import tqdm
def SVM(kernel, decision, x_train, y_train, x_test, y_test, file):
    clf = svm.SVC(kernel=kernel, C=1, decision_function_shape=decision)
    clf.fit(x_train, y_train)

    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)

    acc_train = np.mean(pred_train == y_train)
    acc_test = np.mean(pred_test == y_test)
    print("Kernel:{}\tDecision Function:{}".format(kernel, decision), file=file)
    print("Training Accuracy:\t{}".format(acc_train), file=file)
    print("Testing Accuracy:\t{}\t{}".format(acc_test, clf.score(x_test, y_test)), file=file)
    return acc_train, acc_test


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
    x_train = x_train[train_choice]
    y_train = y_train[train_choice]
    x_test = x_test[test_choice]
    y_test = y_test[test_choice]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    file = open('result.txt', 'w')
    for kernel in tqdm(kernel_list):
        SVM(kernel, 'ovr', x_train, y_train, x_test, y_test, file)
        SVM(kernel, 'ovo', x_train, y_train, x_test, y_test, file)