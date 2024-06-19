# Import packages and set themes ########################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import LanczosCGApproximateGPs as agp
import time
from scipy.sparse.linalg import cg

sns.set_theme


# Standard inversion ####################################################################
sample_size = 5000


start = time.time()

# A = np.zeros((sample_size, sample_size))
# for i in range(sample_size):
#     for j in range(i + 1):
#         A[i, j] = np.random.normal(0, 1)
#         A[j, i] = A[i, j]

A = np.random.normal(0, 1, (sample_size, sample_size))

end = time.time()
print(end - start)


# Start time inverion A
start = time.time()

inverse = np.linalg.inv(A)

end = time.time()
print(end - start)
# end time measurement inversion (n=1000:0.28s, n=5000:4.21s)

# Defining A (O(n^2)) takes the majority of the time, i.e., the inversion (O(n^3)) cannot be standard.


# SVD ###################################################################################

# Start time SVD of A
start = time.time()

U, S, Vh = np.linalg.svd(A, full_matrices = True)

end = time.time()
print(end - start)
# end time measurement defining A (n=1000:0.79s, n=5000:30.95s)

b = np.ones(sample_size)

# Start time SVD of A
start = time.time()

# max_iteration = 100
# cg_actions    = agp.get_conjugate_gradient_actions(A, b, max_iteration)
cg(A, b, maxiter=100)


end = time.time()
print(end - start)
# end time measurement defining A (n=1000:0.79s, n=5000:7.8s)
