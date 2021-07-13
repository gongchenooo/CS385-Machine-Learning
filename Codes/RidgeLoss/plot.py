import numpy as np
import matplotlib.pyplot as plt
import math

lam_list = ['1', '0.1', '0.01', '0.001', '0.0001', '1e-05']
res_dic = {}

def draw_test():
    for lam in lam_list:
        res_dic[lam] = np.load('Results/ten/res_dic_{}.npy'.format(lam), allow_pickle=True).item()['loss_test']
        x = range(len(res_dic[lam]))
    x = [i*10 for i in x]
    plt.clf()
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Testing Loss(Multi Classifier)', fontsize=20)
    for lam in lam_list:
        plt.plot(x, res_dic[lam], label=lam)
    plt.legend(fontsize=15)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    plt.savefig('Testing Loss(Multi Classifier).png')
    plt.show()
def draw_lambda():
    res_dic = []
    distribution = {}
    lam_list = ['1', '0.1', '0.01', '0.001', '0.0001', '1e-05']

    divide = 0.2

    for lam in lam_list:
        tmp_l2 = np.load('Results/ten/res_dic_{}.npy'.format(lam), allow_pickle=True).item()['beta'][0].reshape(-1)
        #tmp_l2_abs = np.square(tmp_l2)
        tmp_l2_abs = abs(tmp_l2)
        dis = {}
        for i in range(tmp_l2_abs.shape[0]):
            try:
                dis[math.log(tmp_l2_abs[i], 10)//divide] += 1
            except KeyError:
                dis[math.log(tmp_l2_abs[i], 10)//divide] = 1
        distribution[lam] = dis

        tmp_l2 = np.sum(np.square(tmp_l2))
        res_dic.append(tmp_l2)

    lam_list_2 = [math.log(float(i), 10) for i in lam_list]
    plt.clf()
    plt.xlabel('log lambda', fontsize=20)
    plt.ylabel('L2-Norm of Parameter', fontsize=20)
    plt.plot(lam_list_2, res_dic, marker='s', ms=10)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig('L2-Norm.png')
    plt.show()

    plt.clf()
    plt.xlabel('log dimension value', fontsize=20)
    plt.ylabel('Number of Dimensions', fontsize=20)
    for lam in lam_list:
        keys = sorted(distribution[lam].keys(),  reverse=True)
        values = [distribution[lam][key] for key in keys]
        plt.plot([i*divide for i in keys], values, label=lam)
    plt.legend(fontsize=15)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig('Distribution of Parameter-l1.png')
    plt.show()

draw_lambda()