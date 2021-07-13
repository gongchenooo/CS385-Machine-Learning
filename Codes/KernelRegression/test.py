import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([10,9,8,7,6,5,4,3,2,1])
print(a.ndim)
gamma = 0.1
c = np.exp(-gamma * np.linalg.norm(a - b) ** 2)
print(c)