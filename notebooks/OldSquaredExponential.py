# Import packages and set themes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import LanczosCGApproximateGPs as agp
import time
from sklearn.gaussian_process.kernels import RBF
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import svds
import scipy.sparse

sns.set_theme()


### Generating data ############################################################

# Parameters
sample_size  = 100
noise_level  = 0.2

# True function
smoothness   = 0.8
def reference_function(x):
    z = np.abs(x + 1)**smoothness - np.abs(x + 3/2)**smoothness
    return z

# Simulating random desing sample
X   = np.random.normal(0, 1, sample_size)
X   = np.sort(X)
X   = np.array([X])
X   = X.transpose()

# Data w/ noise
f0  = reference_function(X[:, 0])
eps = np.random.normal(0, noise_level, sample_size) 
Y   = f0 + eps


### Defining the prior/kernel matrix  ##########################################

# Exponential kernel w/ scaling
length_scale = 4 * sample_size**(-1 / (1 + 2*smoothness))
def sqr_exponential_kernel(x, y):
    z = np.exp(- np.abs(y - x)**2 / length_scale**2) 
    return z

# Computing kernel matrix
kernel = sqr_exponential_kernel

kernel_matrix = np.zeros((sample_size, sample_size)) 
for i in range(0, sample_size):
    for j in range(0, sample_size):
        kernel_matrix[i, j] = kernel(X[i, :], X[j, :])
augmented_kernel_matrix = kernel_matrix + noise_level**2 * np.eye(sample_size)


### True posterior #############################################################

# Inverting the augmented kernel matrix
augmented_inverse = np.linalg.inv(augmented_kernel_matrix)

# Defining the posterior covariance function
def true_posterior_covariance(design_point_1, design_point_2):
    """Returns an the true posterior covariance at the inputs."""

    kernel_vector_at_1 = np.zeros(sample_size)
    for i in range(sample_size):
        kernel_vector_at_1[i] = kernel(X[i, :], design_point_1)

    kernel_vector_at_2 = np.zeros(sample_size)
    for i in range(sample_size):
        kernel_vector_at_2[i] = kernel(X[i, :], design_point_2)

    evaluated_kernel     = kernel(design_point_1, design_point_2)
    covariance_reduction = np.dot(kernel_vector_at_1, augmented_inverse @ kernel_vector_at_2)

    return evaluated_kernel - covariance_reduction

# Evaluating true posterior mean/variance
true_posterior_mean = kernel_matrix @ augmented_inverse @ Y
true_posterior_variance = np.zeros(sample_size)
for i in range(sample_size):
    true_posterior_variance[i] = true_posterior_covariance(X[i, :], X[i, :])

# True credible intervals
true_upper_ci = true_posterior_mean + 2 * np.sqrt(true_posterior_variance)
true_lower_ci = true_posterior_mean - 2 * np.sqrt(true_posterior_variance)

# True posterior plot
fig = plt.figure()
plt.scatter(X[:, 0], Y, color = "gray")
plt.plot(X[:, 0], f0,   color = "black")
plt.plot(X[:, 0], true_posterior_mean, color = "green")
plt.plot(X[:, 0], true_upper_ci,       color = "green")
plt.plot(X[:, 0], true_lower_ci,       color = "green")
plt.ylim((-2, 2))
plt.show()


### Lanczos posterior ##########################################################

# Number of approximation and starting vector
number_of_eigenpairs = 10
starting_vector = Y / np.sqrt(np.sum(Y**2))
# starting_vector = np.random.normal(0, noise_level, sample_size)
# starting_vector = starting_vector / np.sqrt(np.sum(starting_vector**2))

# Collecting Lanczos actions w/ time measurement
lanczos_alg = agp.Lanczos(kernel_matrix, starting_vector, number_of_eigenpairs)
lanczos_alg.run()
lanczos_alg.get_eigenquantities()

lanczos_actions = []
for index in range(number_of_eigenpairs):
    action = lanczos_alg.eigenvectors[:, index]
    lanczos_actions.append(action)

# Running the algorithm
algorithm = agp.Iter_GP(X, Y, noise_level, lanczos_actions, kernel)
algorithm.iter_forward(10)

# Lanczos posterior mean/variance
lanczos_posterior_mean = algorithm.approx_posterior_mean(X)
lanczos_variance_vector = np.zeros(sample_size)
for i in range(sample_size):
    lanczos_variance_vector[i] = algorithm.approx_posterior_covariance(X[i, :], X[i, :])

# Lanczos credible intervals
lanczos_upper_ci = lanczos_posterior_mean + 2 * np.sqrt(lanczos_variance_vector)
lanczos_lower_ci = lanczos_posterior_mean - 2 * np.sqrt(lanczos_variance_vector)

# Lanczos posterior plot
fig = plt.figure()
plt.scatter(X[:, 0], Y, color = "gray")
plt.plot(X[:, 0], f0,   color = "black")
plt.plot(X[:, 0], true_posterior_mean, color = "green")
plt.plot(X[:, 0], true_upper_ci,       color = "green")
plt.plot(X[:, 0], true_lower_ci,       color = "green")
plt.plot(X[:, 0], lanczos_posterior_mean, color = "blue")
plt.plot(X[:, 0], lanczos_upper_ci,       color = "blue")
plt.plot(X[:, 0], lanczos_lower_ci,       color = "blue")
plt.ylim((-2, 2))
plt.show()


# CG-posterior #################################################################

# Number of approximation
max_iteration = 10
cg_actions    = agp.get_conjugate_gradient_actions(augmented_kernel_matrix, Y,
                                                   max_iteration)

# Compute the approximate posterior
algorithm = agp.Iter_GP(X, Y, noise_level, cg_actions, kernel = kernel)
algorithm.iter_forward(10)

# CG posterior mean/variance
cg_posterior_mean = algorithm.approx_posterior_mean(X)
cg_variance_vector = np.zeros(sample_size)
for i in range(sample_size):
    cg_variance_vector[i] = algorithm.approx_posterior_covariance(X[i, :],
                                                                  X[i, :])

# CG credible intervals
cg_upper_ci = cg_posterior_mean + 2 * np.sqrt(cg_variance_vector)
cg_lower_ci = cg_posterior_mean - 2 * np.sqrt(cg_variance_vector)

# Lanczos posterior plot
fig = plt.figure()
plt.scatter(X[:, 0], Y, color = "gray")
plt.plot(X[:, 0], f0,   color = "black")
plt.plot(X[:, 0], true_posterior_mean, color = "green")
plt.plot(X[:, 0], true_upper_ci,       color = "green")
plt.plot(X[:, 0], true_lower_ci,       color = "green")
plt.plot(X[:, 0], cg_posterior_mean, color = "blue")
plt.plot(X[:, 0], cg_upper_ci,       color = "blue")
plt.plot(X[:, 0], cg_lower_ci,       color = "blue")
plt.ylim((-2, 2))
plt.show()


### MSE measurement ############################################################

true_mse      = np.mean((f0 - true_posterior_mean)**2)
lanczos_mse   = np.mean((f0 - lanczos_posterior_mean)**2)
cg_mse        = np.mean((f0 - cg_posterior_mean)**2)
print(true_mse)
print(cg_mse)
print(lanczos_mse)


### Preliminary time measurement ###############################################

# Inverting the augmented kernel matrix w/ time measurement
start = time.time()
augmented_inverse = np.linalg.inv(augmented_kernel_matrix)
end = time.time()
true_time = end - start

# True posterior times (n=1000: 0.087)
print(true_time)

times = np.append(times, true_time)
times

np.mean(times)

times = np.array([])

# Lanczos algorithm w/ time measurement
number_of_eigenpairs = np.sqrt(0.5) * sample_size**(1/(1 + 2 * smoothness)) * np.log(sample_size) / 2
number_of_eigenpairs = number_of_eigenpairs.astype(int) +1
ncv = number_of_eigenpairs + 1

start = time.time()
augmented_kernel_matrix = scipy.sparse.csr_matrix(augmented_kernel_matrix)
svd = svds(augmented_kernel_matrix, k=number_of_eigenpairs, ncv=ncv, v0=Y, maxiter=number_of_eigenpairs)
end = time.time()
lanczos_time = end - start

# Lanczos posterior times (n=1000: 0005)
print(lanczos_time)

# Ratio
print(true_time / lanczos_time)

# Sample size ratio
print(sample_size / number_of_eigenpairs)

sample_sizes =  np.array([5000, 6000,   7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000])
true_times =    np.array([1.18, 1.96,   3.10, 5.03,  7.2, 10.36, 14.23, 19.45, 24.88, 33.10, 41.00])
lanczos_times = np.array([9.22, 13.44, 19.93, 24.5, 31.7, 37.91, 49.75, 64.95, 69.93, 84.30, 96.83])

# Log-log plot
fig = plt.figure()
plt.plot(np.log(sample_sizes), np.log(true_times) - np.log(true_times[0]),         color = "green")
plt.plot(np.log(sample_sizes), np.log(lanczos_times)   - np.log(lanczos_times[0]), color = "blue")
plt.plot(np.log(sample_sizes), np.log(sample_sizes**2) - np.log(5000**2),          color = "black", linestyle = "dashed")
plt.plot(np.log(sample_sizes), np.log(sample_sizes**3) - np.log(5000**3),          color = "black", linestyle = "dashed")
plt.show()






# MSE over sample
true_mse = np.mean((f0 - true_posterior_mean)**2)
print(true_mse)
cg_mse   = np.mean((f0 - cg_posterior_mean)**2)
print(cg_mse)


# More efficient computation of the kernel matrix TODO: Generalize and use this
X   = np.random.normal(0, 1, sample_size)
X   = np.sort(X)

xx, yy = np.meshgrid(X, X)
kernel_matrix = kernel(xx, yy)
augmented_kernel_matrix = kernel_matrix + noise_level**2 * np.eye(sample_size)
kernel_matrix

kernel = RBF(length_scale)
kernel_matrix = kernel(X)
augmented_kernel_matrix = kernel_matrix + noise_level**2 * np.eye(sample_size)


