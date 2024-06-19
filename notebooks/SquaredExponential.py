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
sample_size  = 5000
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

length_scale = 4 * sample_size**(-1 / (1 + 2*smoothness))
kernel = RBF(length_scale)
kernel_matrix = kernel(X, X)
augmented_kernel_matrix = kernel_matrix + noise_level**2 * np.eye(sample_size)


### True posterior #############################################################

# Inverting the augmented kernel matrix
augmented_inverse = np.linalg.inv(augmented_kernel_matrix)

# Defining the posterior covariance function
def true_posterior_covariance(design_point_1, design_point_2):
    """Returns an the true posterior covariance at the inputs."""

    kernel_vector_at_1 = kernel(X, np.array([design_point_1]))
    kernel_vector_at_2 = kernel(X, np.array([design_point_2]))

    evaluated_kernel     = kernel(np.array([design_point_1]),
                                  np.array([design_point_2]))
    covariance_reduction = kernel_vector_at_1.transpose() @ \
                           augmented_inverse @ kernel_vector_at_2

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
plt.scatter(X[:, 0], Y, color = "lightgray")
plt.plot(X[:, 0], true_posterior_mean, color = "green", label='True posterior')
plt.plot(X[:, 0], true_upper_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], true_lower_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], f0,   color = "black", label='True function '+ r"$f_0$")
plt.ylim((-1, 1))
plt.legend(loc = 'upper right')
plt.show()

true_mse = np.mean((f0 - true_posterior_mean)**2)
# 6*10^-4


### Empirical eigenvector posterior ############################################

number_of_eigenpairs = 80
U, S, Vh = np.linalg.svd(kernel_matrix)

eigenvector_actions = []
for index in range(number_of_eigenpairs):
    action = U[:, index]
    eigenvector_actions.append(action)

# Running the algorithm
algorithm = agp.Iter_GP(X, Y, noise_level, eigenvector_actions, kernel)
algorithm.iter_forward(80)

# Eigenvector posterior mean/variance
eigenvector_posterior_mean = algorithm.approx_posterior_mean(X)
eigenvector_variance_vector = np.zeros(sample_size)
for i in range(sample_size):
    eigenvector_variance_vector[i] = algorithm.approx_posterior_covariance(X[i, :], X[i, :])

# Eigenvector credible intervals
eigenvector_upper_ci = eigenvector_posterior_mean + 2 * np.sqrt(eigenvector_variance_vector)
eigenvector_lower_ci = eigenvector_posterior_mean - 2 * np.sqrt(eigenvector_variance_vector)

# Eigenvector posterior plot
fig = plt.figure()
plt.scatter(X[:, 0], Y, color = "lightgray")
plt.plot(X[:, 0], true_posterior_mean, color = "green", label='True posterior')
plt.plot(X[:, 0], true_upper_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], true_lower_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], eigenvector_posterior_mean, color = "blue", label='EVGP posterior')
plt.plot(X[:, 0], eigenvector_upper_ci,       color = "blue", linestyle = 'dashed')
plt.plot(X[:, 0], eigenvector_lower_ci,       color = "blue", linestyle = 'dashed')
plt.plot(X[:, 0], f0,   color = "black", label='True function '+ r"$f_0$")
plt.ylim((-1, 1))
plt.legend(loc = 'upper right')
plt.show()

eigenvector_mse = np.mean((f0 - eigenvector_posterior_mean)**2)
# m=80: 6*10^-4


### Lanczos posterior ##########################################################

# Number of approximation and starting vector
number_of_eigenpairs = 20
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

lanczos_alg.eigenvalues

# Running the algorithm
algorithm = agp.Iter_GP(X, Y, noise_level, lanczos_actions, kernel)
algorithm.iter_forward(3)

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
plt.scatter(X[:, 0], Y, color = "lightgray")
plt.plot(X[:, 0], true_posterior_mean, color = "green", label='True posterior')
plt.plot(X[:, 0], true_upper_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], true_lower_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], lanczos_posterior_mean, color = "blue", label='LGP posterior')
plt.plot(X[:, 0], lanczos_upper_ci,       color = "blue", linestyle = 'dashed')
plt.plot(X[:, 0], lanczos_lower_ci,       color = "blue", linestyle = 'dashed')
plt.plot(X[:, 0], f0,   color = "black", label='True function '+ r"$f_0$")
plt.ylim((-1, 1))
plt.legend(loc = 'upper right')
plt.show()


# CG-posterior #################################################################

# Number of approximation
max_iteration = 320
cg_actions    = agp.get_conjugate_gradient_actions(augmented_kernel_matrix, Y,
                                                   max_iteration)
print("actions done")

# Compute the approximate posterior
algorithm = agp.Iter_GP(X, Y, noise_level, cg_actions, kernel = kernel)
algorithm.iter_forward(160)
print("algorithm done")

# CG posterior mean/variance
cg_posterior_mean = algorithm.approx_posterior_mean(X)
cg_variance_vector = np.zeros(sample_size)
for i in range(sample_size):
    cg_variance_vector[i] = algorithm.approx_posterior_covariance(X[i, :],
                                                                  X[i, :])
# CG credible intervals
cg_upper_ci = cg_posterior_mean + 2 * np.sqrt(cg_variance_vector)
cg_lower_ci = cg_posterior_mean - 2 * np.sqrt(cg_variance_vector)

# CG posterior plot
fig = plt.figure()
plt.scatter(X[:, 0], Y, color = "lightgray")
plt.plot(X[:, 0], true_posterior_mean, color = "green", label='True posterior')
plt.plot(X[:, 0], true_upper_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], true_lower_ci,       color = "green", linestyle = 'dashed')
plt.plot(X[:, 0], cg_posterior_mean, color = "blue", label='CGGP posterior')
plt.plot(X[:, 0], cg_upper_ci,       color = "blue", linestyle = 'dashed')
plt.plot(X[:, 0], cg_lower_ci,       color = "blue", linestyle = 'dashed')
plt.plot(X[:, 0], f0,   color = "black", label='True function '+ r"$f_0$")
plt.ylim((-1, 1))
plt.xlabel("   ") # In order to align plot properly
plt.legend(loc = 'upper right')
plt.show()


cg_mse = np.mean((f0 - cg_posterior_mean)**2)
# m=40:  0.01
# m=80:  6*10^-4
# m=160: 6*10^-4


### Computation for the table ###################################################

number_of_runs = 100
true_mse_vector = np.array([])
coverage_vector = np.zeros(5000)

for run in range(number_of_runs):

    # Generate data
    sample_size  = 5000
    noise_level  = 0.2

    smoothness   = 0.8
    def reference_function(x):
        z = np.abs(x + 1)**smoothness - np.abs(x + 3/2)**smoothness
        return z

    X   = np.random.normal(0, 1, sample_size)
    X   = np.sort(X)
    X   = np.array([X])
    X   = X.transpose()

    f0  = reference_function(X[:, 0])
    eps = np.random.normal(0, noise_level, sample_size) 
    Y   = f0 + eps

    # Kernel matrix
    length_scale = 4 * sample_size**(-1 / (1 + 2*smoothness))
    kernel = RBF(length_scale)
    kernel_matrix = kernel(X, X)
    augmented_kernel_matrix = kernel_matrix + noise_level**2 * np.eye(sample_size)
    augmented_inverse = np.linalg.inv(augmented_kernel_matrix)

    # Defining the posterior covariance function
    def true_posterior_covariance(design_point_1, design_point_2):
        """Returns an the true posterior covariance at the inputs."""

        kernel_vector_at_1 = kernel(X, np.array([design_point_1]))
        kernel_vector_at_2 = kernel(X, np.array([design_point_2]))

        evaluated_kernel     = kernel(np.array([design_point_1]),
                                      np.array([design_point_2]))
        covariance_reduction = kernel_vector_at_1.transpose() @ \
                               augmented_inverse @ kernel_vector_at_2

        return evaluated_kernel - covariance_reduction

    # Evaluating true posterior mean/variance
    true_posterior_mean = kernel_matrix @ augmented_inverse @ Y

    true_posterior_variance = np.zeros(sample_size)
    for i in range(sample_size):
        true_posterior_variance[i] = true_posterior_covariance(X[i, :], X[i, :])

    # True credible intervals
    true_upper_ci = true_posterior_mean + 2 * np.sqrt(true_posterior_variance)
    true_lower_ci = true_posterior_mean - 2 * np.sqrt(true_posterior_variance)

    # Evaluating mse and coverage
    true_mse        = np.mean((f0 - true_posterior_mean)**2)
    true_mse_vector = np.append(true_mse_vector, true_mse)

    coverage        = ( true_lower_ci <= f0 ) * ( f0 <= true_upper_ci ) 
    coverage_vector = coverage_vector + coverage

np.mean(true_mse_vector)
np.std(true_mse_vector)
coverage_vector



### Time measurement ###############################################

# Noise level
noise_level  = 0.2

# True function
smoothness   = 0.8
def reference_function(x):
    z = np.abs(x + 1)**smoothness - np.abs(x + 3/2)**smoothness
    return z

sample_sizes = np.array([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,
                         13000, 14000, 15000])
true_times = np.array([])
cg_times   = np.array([])

for sample_size in sample_sizes:

    # Simulating random desing sample
    X   = np.random.uniform(0, 1, sample_size)
    X   = np.sort(X)
    X   = np.array([X])
    X   = X.transpose()

    # Data w/ noise
    f0  = reference_function(X[:, 0])
    eps = np.random.normal(0, noise_level, sample_size) 
    Y   = f0 + eps

    # Simulating random desing sample
    X   = np.random.uniform(0, 1, sample_size)
    X   = np.sort(X)
    X   = np.array([X])
    X   = X.transpose()

    # Compute kernel matrix
    length_scale = 4 * sample_size**(-1 / (1 + 2*smoothness))
    kernel = RBF(length_scale)
    kernel_matrix = kernel(X, X)
    augmented_kernel_matrix = kernel_matrix + noise_level**2 * np.eye(sample_size)

    # Inverting the augmented kernel matrix w/ time measurement
    start = time.time()
    augmented_inverse = np.linalg.inv(augmented_kernel_matrix)
    end = time.time()
    true_time = end - start
    true_times = np.append(true_times, true_time)
    print(true_times)

    # Computing the CG actions w/ time measurement

    max_iteration = 2**(-1/2) * sample_size**(1 / (2*smoothness + 1) ) * \
                    np.log(sample_size)
    max_iteration = int(max_iteration)

    start = time.time()
    cg_actions = agp.get_conjugate_gradient_actions(augmented_kernel_matrix, Y,
                                                    max_iteration)
    end = time.time()
    cg_time = end - start
    cg_times = np.append(cg_times, cg_time)
    print(cg_times)

# Saving the data for later
sample_sizes = np.array([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,
                         13000, 14000, 15000])
true_times   = np.array([1.11965728, 2.02679276, 3.09848332, 4.91819, 6.5640049,
                         9.29445887, 11.84166908, 15.8407321, 19.42188263,
                         24.92394805, 30.04538846])
cg_times     = np.array([9.20516801, 14.61769223, 22.171592, 29.81479168,
                         40.51051331, 51.57282472, 63.49643707, 76.99794006,
                         88.53071332, 104.84900212, 122.37278628])

# Log-log plot
fig = plt.figure()
plt.plot(np.log(sample_sizes), np.log(true_times) - np.log(true_times[0]),
         color="green", label="True posterior")
plt.plot(np.log(sample_sizes), np.log(sample_sizes**3) - np.log(5000**3),
         color="green", linestyle="dashed", label=r"$n^{3}$")
plt.plot(np.log(sample_sizes), np.log(cg_times) - np.log(cg_times[0]),
         color="blue", label="CGGP posterior")
plt.plot(np.log(sample_sizes), np.log(sample_sizes**2) - np.log(5000**2),
         color = "blue", linestyle = "dashed", label=r"$n^2$")
plt.xlabel(r"$log(n)$")
plt.ylabel(r"$log(\text{time})$")
plt.legend()
plt.show()
