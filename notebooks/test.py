import numpy as np
import LanczosCGApproximateGPs as gp

sample_size = 5
max_iter = 3
noise_level = 1

X = np.array([np.linspace(0, 1, sample_size)])
X = X.transpose()
Y = X[:, 0]**2
print(Y)

actions = []
index = 0
for iter in range(0, max_iter):
    action = np.zeros(sample_size)
    action[index] = 1
    actions.append(action)
    index = index + 1

def gaussian_kernel(x, y):
    z = np.minimum(x, y)
    return z

algorithm = gp.Iter_GP(X, Y, noise_level, actions, kernel = gaussian_kernel)

x = np.array([0, 1, 2])
y = np.array([1, 0, 2])
algorithm.kernel(x, 0)
