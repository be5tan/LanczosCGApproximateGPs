import numpy as np
import LanczosCGApproximateGPs as agp

class CG_GP():
    """
    Performance version of the conjugate gradient approximate GP.
    """
    
    def __init__(self, design_matrix, response_vector, noise_level, kernel, approx_number):
        # User input
        self.design_matrix   = design_matrix
        self.response_vector = response_vector
        self.noise_level     = noise_level
        self.kernel          = kernel
        self.approx_number   = approx_number

        # Model parameters
        self.sample_size   = np.shape(design_matrix)[0]
        self.sample_dim    = np.shape(design_matrix)[1]

        self.kernel_matrix = np.zeros((self.sample_size, self.sample_size)) 
        for i in range(0, self.sample_size):
            for j in range(0, self.sample_size):
                self.kernel_matrix[i, j] = self.kernel(self.design_matrix[i, :], self.design_matrix[j, :])

        self.augmented_kernel_matrix = self.kernel_matrix + noise_level**2 * np.eye(self.sample_size)

        self.actions = agp.get_conjugate_gradient_actions(self.augmented_kernel_matrix, self.response_vector, self.approx_number)
        self.conjugate_actions = []
        self.weights = []

        self.approx_augmented_inverse = np.zeros((self.sample_size, self.sample_size))

        for j in range(self.approx_number):
            action           = self.actions[j]

            conjugate_action = action - self.approx_augmented_inverse @ self.augmented_kernel_matrix @ action
            self.conjugate_actions.append(conjugate_action)

            weight = np.dot(action, self.augmented_kernel_matrix @ conjugate_action) 
            self.weights.append(weight)

            self.approx_augmented_inverse   = self.approx_augmented_inverse + rank_one_update / weight
        
        self.approx_representer_weights = self.approx_augmented_inverse @ self.response_vector
            

    def approx_posterior_mean(self, input_array):
        """Returns an approximate version of the posterior mean at the inputs."""
        input_size         = np.shape(input_array)[0]
        approx_mean_vector = np.zeros(input_size)

        for j in range(input_size):
            kernel_vector = np.zeros(self.sample_size)
            for i in range(self.sample_size):
                kernel_vector[i] = self.kernel(self.design_matrix[i, :], input_array[j, :])

            approx_mean_vector[j] = np.dot(kernel_vector, self.approx_representer_weights) 

        return approx_mean_vector


    def approx_posterior_covariance(self, design_point_1, design_point_2):
        """Returns an approximate version of the posterior covariance at the inputs."""
        kernel_vector_at_1 = np.zeros(self.sample_size)
        for i in range(self.sample_size):
            kernel_vector_at_1[i] = self.kernel(self.design_matrix[i, :], design_point_1)

        kernel_vector_at_2 = np.zeros(self.sample_size)
        for i in range(self.sample_size):
            kernel_vector_at_2[i] = self.kernel(self.design_matrix[i, :], design_point_2)

        evaluated_kernel     = self.kernel(design_point_1, design_point_2)
        covariance_reduction = np.dot(kernel_vector_at_1, self.approx_augmented_inverse @ kernel_vector_at_2)

        return evaluated_kernel - covariance_reduction















