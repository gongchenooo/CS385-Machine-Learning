import matplotlib.pyplot as plt
import numpy as np

res_binary = np.load('Results/binary/res_dic_binary.npy', allow_pickle=True).item()
res_ten = np.load('Results/ten/res_dic.npy', allow_pickle=True).item()
print(res_binary.keys())
print(res_ten.keys())


loss_train_binary = res_binary['loss_train']
loss_test_binary = res_binary['loss_test']
loss_train_ten = res_ten['loss_train']
loss_test_ten = res_ten['loss_test']
x = range(0, len(loss_test_ten))
x = [i*10 for i in x]
print(x)
plt.clf()
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.plot(x, loss_train_binary, label="Training Loss(ten binary-classifier)", color='r', linestyle='-')
plt.plot(x, loss_test_binary, label="Testing Loss(ten binary-classifier)", color='b', linestyle='-')
plt.xticks(size = 20)
plt.yticks(size = 20)
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
plt.xticks(size = 20)
plt.yticks(size = 20)
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
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.legend(fontsize=15)
plt.savefig('Accuracy.png')
plt.show()

plt.clf()
print(res_binary['acc_class_train'][-1])
print(res_ten['acc_class_train'][-1])

x = np.arange(10)
a = [res_binary['acc_class_train'][-1][i][0] / res_binary['acc_class_train'][-1][i][1] for i in range(10)]
b = [res_ten['acc_class_train'][-1][i][0] / res_ten['acc_class_train'][-1][i][1] for i in range(10)]
total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='binary-classifier')
plt.bar(x + width, b, width=width, label='multi-classifier')
plt.xlabel('Number Class', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.savefig('Training Accuracy of Different Classes.png')
plt.show()

plt.clf()
print(res_binary['acc_class_test'][-1])
print(res_ten['acc_class_test'][-1])

x = np.arange(10)
a = [res_binary['acc_class_test'][-1][i][0] / res_binary['acc_class_test'][-1][i][1] for i in range(10)]
b = [res_ten['acc_class_test'][-1][i][0] / res_ten['acc_class_test'][-1][i][1] for i in range(10)]
total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2
plt.xlabel('Number Class', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.bar(x, a,  width=width, label='binary-classifier')
plt.bar(x + width, b, width=width, label='multi-classifier')
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend(fontsize=15)
plt.savefig('Testing Accuracy of Different Classes.png')
plt.show()