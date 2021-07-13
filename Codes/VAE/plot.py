import numpy as np
import matplotlib.pyplot as plt

KLD = np.load('KLD_loss.npy')
loss = np.load('loss.npy')
rec_loss = np.load('rec_loss.npy')

plt.clf()
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
x = range(len(loss))
plt.plot(x, loss, label='Loss')
#plt.plot(x, rec_loss, label='reconstruction loss', color='r')
#plt.plot(x, KLD, label='KL Divergence Loss', color='yellow')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('kl_loss.png')
plt.show()