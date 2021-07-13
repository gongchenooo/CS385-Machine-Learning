from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold, datasets

# 4096 3072 1024
layer = 12
X = np.load("data/x_AlexNet_{}.npy".format(layer))[2000:4000]
y = np.load("data/y_AlexNet_{}.npy".format(layer))[2000:4000]
print(X[0].shape)
exit()
pca = PCA(n_components=2)
res = pca.fit_transform(X)
label = y
fig = plt.figure()
ax = fig.gca()
ax.scatter(res[:, 0], res[:, 1], c=label, cmap=plt.cm.Spectral)
plt.savefig('pca_{}.png'.format(layer))
plt.show()


'''tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(X)  # 转换后的输出
fig = plt.figure(figsize=(8, 8))


plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)


plt.axis('tight')
plt.savefig('tsne_{}.png'.format(layer))
plt.show()'''