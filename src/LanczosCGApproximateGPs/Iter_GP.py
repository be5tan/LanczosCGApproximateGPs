import numpy as np

class Iter_GP():
    """
    Approximate GP iteration.

    Parameters
    ----------
    design_matrix: array
        nxd-dim design matrix.

    response_vector: array
        n-dim vector of observed data.

    noise_level: float
        standard deviation of the regression model.

    actions: list of arrays
        list of n-dim vectors determining the search directions of the algorithm.

    kernel: function
        k: R^dxd -> R, positive definite kernel function.

    Attributes
    ----------
    iter: float
        Number of iterations that have been performed by the algorithm.

    sample_size: float
        Sample size of the regression model.

    sample_dim: float
        Dimension of one observation of the regression model.

    approx_representer_weights: array
        Approximation of the n-dim weights representing the posterior mean.

    approx_augmented_inverse: array
        Low rank nxn-matrix approximating the inverse of the augmented kernel matrix.

    Methods
    -------
    kernel()

    """
    def __init__(self, design_matrix, response_vector, noise_level, actions, kernel):
        # User input
        self.design_matrix   = design_matrix
        self.response_vector = response_vector
        self.noise_level     = noise_level
        self.actions         = actions
        self.kernel          = kernel

        # Model parameters
        self.sample_size   = np.shape(design_matrix)[0]
        self.sample_dim    = np.shape(design_matrix)[1]

        self.kernel_matrix = np.zeros((self.sample_size, self.sample_size)) 
        for i in range(0, self.sample_size):
            for j in range(0, self.sample_size):
                self.kernel_matrix[i, j] = self.kernel(self.design_matrix[i, :], self.design_matrix[j, :])

        self.augmented_kernel_matrix = self.kernel_matrix + noise_level**2 * np.eye(self.sample_size)

        # Initialize quantities of the iteration
        self.iter                       = 0
        self.approx_representer_weights = np.zeros(self.sample_size)
        self.approx_augmented_inverse   = np.zeros((self.sample_size, self.sample_size))

    def iter_forward(self, iter_num):
        """Performs iter_num iterations of the Iter_GP algorithm."""
        for iter in range(iter_num):
            self.__iter_forward_one()


    def __iter_forward_one(self):
        """Performs one iteration of the Iter_GP algorithm."""
        action           = self.actions[self.iter]
        conjugate_action = action - self.approx_augmented_inverse @ self.augmented_kernel_matrix @ action
        weight           = np.dot(action, self.augmented_kernel_matrix @ conjugate_action) 

        rank_one_update = np.zeros((self.sample_size, self.sample_size))
        for i in range(0, self.sample_size):
            for j in range(0, self.sample_size):
                rank_one_update[i, j] = conjugate_action[i] * conjugate_action[j]

        vector_update = np.dot(conjugate_action, self.response_vector) * self.response_vector

        self.approx_augmented_inverse   = self.approx_augmented_inverse   + rank_one_update / weight
        self.approx_representer_weights = self.approx_representer_weights + vector_update   / weight 

        self.iter = self.iter + 1

    # def mean()
    #     """
    #     """

    # def covariance()
    #     """
    #     """
