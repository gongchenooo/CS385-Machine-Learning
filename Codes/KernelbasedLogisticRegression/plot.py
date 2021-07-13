import matplotlib.pyplot as plt
import numpy as np
def plot_rbf():
    res_dic = {}
    sigma_list = [0.1, 0.5, 1, 5, 10]
    for sigma in sigma_list:
        res_dic[sigma] = np.load('Results/rbf/res_binary_{}.npy'.format(sigma), allow_pickle=True).item()['loss_test']
        x = range(len(res_dic[sigma]))
    x = [i*2 for i in x]
    plt.clf()
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Testing Loss(Binary Classifier)', fontsize=20)
    for sigma in sigma_list:
        plt.plot(x, res_dic[sigma], label=sigma)
    plt.legend(fontsize=15)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    plt.savefig('Testing Loss(Binary Classifier).png')
    plt.show()

def plot_poly():
    res_dic = {}
    d_list = [1,2,3,4]
    for d in d_list:
        res_dic[d] = np.load('Results/poly/res_ten_{}.npy'.format(d), allow_pickle=True).item()['loss_test']
        x = range(len(res_dic[d]))
    x = [i*2 for i in x]
    plt.clf()
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Testing Loss(Multi Classifier)', fontsize=20)
    for sigma in d_list:
        plt.plot(x, res_dic[sigma], label=sigma)
    plt.legend(fontsize=15)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    plt.savefig('Testing Loss(Multi Classifier).png')
    plt.show()
def plot_cos():
    res_binary = np.load('Results/cos/res_binary.npy', allow_pickle=True).item()
    res_ten = np.load('Results/cos/res_ten.npy', allow_pickle=True).item()
    print(res_binary.keys())
    print(res_ten.keys())

    loss_train_binary = res_binary['loss_train']
    loss_test_binary = res_binary['loss_test']
    loss_train_ten = res_ten['loss_train']
    loss_test_ten = res_ten['loss_test']
    x = range(0, len(loss_test_ten))
    x = [i * 10 for i in x]
    print(x)
    plt.clf()
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.plot(x, loss_train_binary, label="Training Loss(ten binary-classifier)", color='r', linestyle='-')
    plt.plot(x, loss_test_binary, label="Testing Loss(ten binary-classifier)", color='b', linestyle='-')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig('Loss_binary.png')
    plt.show()

    plt.clf()
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.plot(x, loss_train_ten, label="Training Loss(multi-classifier)", color='r', linestyle='-')
    plt.plot(x, loss_test_ten, label="Testing Loss(multi-classifier)", color='b', linestyle='-')
    plt.tight_layout()
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=15)
    plt.savefig('Loss_ten.png')
    plt.show()

    plt.clf()
    loss_train_binary = res_binary['accuracy_train']
    loss_test_binary = res_binary['accuracy_test']
    loss_train_ten = res_ten['accuracy_train']
    loss_test_ten = res_ten['accuracy_test']
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(x, loss_train_binary, label="Training Acc(ten binary-classifier)", color='r', linestyle='--')
    plt.plot(x, loss_test_binary, label="Testing Acc(ten binary-classifier)", color='b', linestyle='--')
    plt.plot(x, loss_train_ten, label="Training Acc(multi-classifier)", color='r', linestyle='-')
    plt.plot(x, loss_test_ten, label="Testing Acc(multi-classifier)", color='b', linestyle='-')
    plt.tight_layout()
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=15)
    plt.savefig('Accuracy.png')
    plt.show()

def plot_sigmoid():
    res_dic = {}
    a_list = [0.5, 1, 2, 3]
    for a in a_list:
        res_dic[a] = np.load('Results/sigmoid/res_ten_{}_0.npy'.format(a), allow_pickle=True).item()['loss_test']
        x = range(len(res_dic[a]))
    x = [i*2 for i in x]
    plt.clf()
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Testing Loss(Multi Classifier)', fontsize=20)
    for sigma in a_list:
        plt.plot(x, res_dic[sigma], label=sigma)
    plt.legend(fontsize=15)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    plt.savefig('Testing Loss(Multi Classifier).png')
    plt.show()
plot_sigmoid()
exit()
import math
def plot_parameters():
    res_dic = []
    sigma_list = [1,2,3,4]
    distribution = {}
    divide = 0.2
    for sigma in sigma_list:
        tmp_l2 = np.load('Results/poly/res_ten_{}.npy'.format(sigma), allow_pickle=True).item()['beta'][-1].reshape(-1)
        # tmp_l2_abs = np.square(tmp_l2)
        tmp_l2_abs = abs(tmp_l2)
        dis = {}
        for i in range(tmp_l2_abs.shape[0]):
            try:
                dis[math.log(tmp_l2_abs[i], 10) // divide] += 1
            except KeyError:
                dis[math.log(tmp_l2_abs[i], 10) // divide] = 1
        distribution[sigma] = dis

        tmp_l2 = np.sum(tmp_l2_abs)
        res_dic.append(tmp_l2)

    plt.clf()
    plt.xlabel('log dimension value', fontsize=20)
    plt.ylabel('Number of Dimensions', fontsize=20)
    for sigma in sigma_list:
        keys = sorted(distribution[sigma].keys(), reverse=True)
        values = [distribution[sigma][key] for key in keys]
        plt.plot([i * divide for i in keys], values, label=sigma)
    plt.legend(fontsize=15)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig('Distribution of Parameter-l1.png')
    plt.show()



#plot_parameters()
#plot_rbf()