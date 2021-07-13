import numpy as np
import matplotlib.pyplot as plt

res = np.load('Results/binary/res_dic.npy', allow_pickle=True).item()
print(res[1])
plt.clf()
x = range(1, 26)
acc_list = []
plt.xlabel('k', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
for key in range(1, 26):
    acc_list.append(res[key]['acc'])
plt.plot(x, acc_list, marker='*', color='r')
plt.savefig('accuracy_2.png')
plt.show()
#print(res)