import numpy as np
import matplotlib.pyplot as plt

def plot_data():
    x_train = np.load('../Data/train_x_1.npy')
    y_train = np.load('../Data/train_y_1.npy')
    x_test = np.load('../Data/test_x_1.npy')
    y_test = np.load('../Data/test_y_1.npy')


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

    x_train_dic = {i: [] for i in range(10)}
    x_test_dic = {i: [] for i in range(10)}

    for i in range(x_train.shape[0]):
        x_train_dic[y_train[i]].append(x_train[i])
    for i in range(x_test.shape[0]):
        x_test_dic[y_test[i]].append(x_test[i])
    for key in x_train_dic.keys():
        x_train_dic[key] = np.array(x_train_dic[key])
        x_test_dic[key] = np.array(x_test_dic[key])
        x_train_dic[key] = (x_train_dic[key].mean(axis=0), x_train_dic[key].std(axis=0))
        x_test_dic[key] = (x_test_dic[key].mean(axis=0), x_test_dic[key].std(axis=0))
    for key in range(10):
        print(np.mean(abs(x_train_dic[key][1])), np.mean(abs(x_test_dic[key][1])))
        print(key, np.mean(abs(x_train_dic[key][0] - x_test_dic[key][0])) / np.max(abs(x_train_dic[key][0])))
        print(key, np.mean(abs(x_train_dic[key][1] - x_test_dic[key][1])) / np.max(abs(x_train_dic[key][1])))

    x = range(10)
    plt.plot(x, [np.mean(abs(x_train_dic[key][0] - x_test_dic[key][0])) / np.max(abs(x_train_dic[key][0])) for key in range(10)], label='mean difference', marker='o')
    plt.plot(x, [np.mean(abs(x_train_dic[key][1] - x_test_dic[key][1])) / np.max(abs(x_train_dic[key][1])) for key in range(10)], label='variance difference', marker='*', ms=10)
    plt.xlabel('Digit Class', fontsize=20)
    plt.ylabel('Relative Difference', fontsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig('Data Distribution Difference')
    plt.show()
    exit()


    classes = {i: 0 for i in range(10)}
    for i in y_train:
        classes[i] += 1
    sum1 = sum(classes.values())
    for key in classes.keys():
        classes[key] /= sum1
    labels = classes.keys()
    x = classes.values()
    plt.pie(x=x, labels=labels, labeldistance=1.1, autopct='%1.1f%%',
            radius=1.2, wedgeprops={'linewidth': '1.5', 'edgecolor': 'black'},
            textprops={'fontsize': 10, 'color': 'black'})
    plt.title("Training Data Distribution", fontsize=15)
    plt.tight_layout()
    plt.savefig('Training Data Distribution')
    plt.show()

    plt.clf()
    classes = {i:0 for i in range(10)}
    for i in y_test:
        classes[i] += 1
    sum1 = sum(classes.values())
    for key in classes.keys():
        classes[key] /= sum1
    labels = classes.keys()
    x = classes.values()
    plt.pie(x=x, labels=labels, labeldistance=1.1,autopct='%1.1f%%',
        radius=1.2,wedgeprops={'linewidth':'1.5','edgecolor':'black'},
        textprops={'fontsize':10,'color':'black'})
    plt.title("Testing Data Distribution", fontsize=15)
    plt.tight_layout()
    plt.savefig('Testing Data Distribution')
    plt.show()


plot_data()