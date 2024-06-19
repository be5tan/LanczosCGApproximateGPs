################################################################################
### Computations for the table #################################################
################################################################################

# Import packages and set themes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import LanczosCGApproximateGPs as agp
import time
from sklearn.gaussian_process.kernels import RBF
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import svds
import scipy.sparse

sns.set_theme()


### True posterior data ########################################################

number_of_runs = 10
mse_vector = np.array([])
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
    mse        = np.mean((f0 - true_posterior_mean)**2)
    mse_vector = np.append(true_mse_vector, true_mse)

    coverage        = ( true_lower_ci <= f0 ) * ( f0 <= true_upper_ci ) 
    coverage_vector = coverage_vector + coverage

true_mse_vector
np.mean(true_mse_vector)
np.std(true_mse_vector)
coverage_vector = pd.Series(coverage_vector)
coverage_vector.describe()


### EVGP posterior data ########################################################

number_of_runs = 10
mse_vector = np.array([])
coverage_vector = np.zeros(1000)

for run in range(number_of_runs):

    # Generate data
    sample_size  = 1000
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

    print(run)

    # Kernel matrix
    length_scale = 4 * sample_size**(-1 / (1 + 2*smoothness))
    kernel = RBF(length_scale)
    kernel_matrix = kernel(X, X)
    augmented_kernel_matrix = kernel_matrix + noise_level**2 * np.eye(sample_size)

    print(run)

    number_of_eigenpairs = 80
    U, S, Vh = np.linalg.svd(kernel_matrix)

    eigenvector_actions = []
    for index in range(number_of_eigenpairs):
        action = U[:, index]
        eigenvector_actions.append(action)

    print(run)

    # Running the algorithm
    algorithm = agp.Iter_GP(X, Y, noise_level, eigenvector_actions, kernel)
    algorithm.iter_forward(80)

    print(run)

    # Eigenvector posterior mean/variance
    eigenvector_posterior_mean = algorithm.approx_posterior_mean(X)
    eigenvector_variance_vector = np.zeros(sample_size)
    for i in range(sample_size):
        eigenvector_variance_vector[i] = algorithm.approx_posterior_covariance(X[i, :], X[i, :])

    # Eigenvector credible intervals
    eigenvector_upper_ci = eigenvector_posterior_mean + 2 * np.sqrt(eigenvector_variance_vector)
    eigenvector_lower_ci = eigenvector_posterior_mean - 2 * np.sqrt(eigenvector_variance_vector)

    # Evaluating mse and coverage
    mse        = np.mean((f0 - eigenvector_posterior_mean)**2)
    mse_vector = np.append(mse_vector, mse)

    coverage        = ( eigenvector_lower_ci <= f0 ) * ( f0 <= eigenvector_upper_ci ) 
    coverage_vector = coverage_vector + coverage

    print(run)

mse_vector
np.mean(mse_vector)
np.std(mse_vector)
coverage_vector = pd.Series(coverage_vector)
coverage_vector.describe()
