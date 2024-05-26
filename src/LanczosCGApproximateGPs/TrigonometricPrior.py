import numpy as np

class TrigonometricPrior():
    """
    Trigonometric random series prior on [-pi, pi].

    **Parameters**

    *eigenvalues*: ``array``. Array of eigenvalues of the kernel operator corresponding to the random series prior. The number of eigenvalues has to be odd.

    **Methods**
    +--------------------+-----------------------------------------+
    | evaluate( ``points`` )  | Evaluates the prior at a points x. |
    +--------------------+-----------------------------------------+
    """

    def __init__(self, eigenvalues):
        self.eigenvalues           = eigenvalues
        self.number_of_eigenvalues = self.eigenvalues.shape[0]
        if self.number_of_eigenvalues % 2 == 0:
            raise Exception(f"Number of eigenvalues should be odd. ({self.number_of_eigenvalues=})")

        sqrt_eigenvalues       = np.sqrt(self.eigenvalues) 
        self.const_coefficient = sqrt_eigenvalues[0] # * np.random.normal(0, 1) 

        self.half_number_of_eigenvalues = np.floor(self.number_of_eigenvalues / 2).astype(int)

        if self.half_number_of_eigenvalues != 0:
            reduced_sqrt_eigenvalues = np.delete(sqrt_eigenvalues, 0)
            self.cos_coefficients    = reduced_sqrt_eigenvalues[0::2] * \
                                       np.random.normal(0, 1, self.half_number_of_eigenvalues)
            self.sin_coefficients    = reduced_sqrt_eigenvalues[1::2] * \
                                       np.random.normal(0, 1, self.half_number_of_eigenvalues)

            # For testing comment out the lines including randomness above and use these instead
            # self.cos_coefficients    = reduced_sqrt_eigenvalues[0::2] # * \
            #                            # np.random.normal(0, 1, self.half_number_of_eigenvalues)
            # self.sin_coefficients    = reduced_sqrt_eigenvalues[1::2] # * \
                                       # np.random.normal(0, 1, self.half_number_of_eigenvalues)

    def evaluate(self, points):
        """
        **Parameters**

        *point*: ``float or array``. Points from [-pi, pi] at which to evaluate the prior.
        """
        if type(points) == float or type(points) == int:
            number_of_points  = 1
        else:
            number_of_points  = points.size

        const_evaluations = np.ones(number_of_points) / np.sqrt(2 * np.pi)  
        evaluated_prior   = const_evaluations * self.const_coefficient

        if self.half_number_of_eigenvalues != 0:
            cos_evaluations = np.zeros((number_of_points, self.half_number_of_eigenvalues))
            sin_evaluations = np.zeros((number_of_points, self.half_number_of_eigenvalues))

            for index in range(self.half_number_of_eigenvalues):
                cos_evaluations[:, index] = np.cos((index + 1) * points) / np.sqrt(np.pi)
                sin_evaluations[:, index] = np.sin((index + 1) * points) / np.sqrt(np.pi)

            evaluated_prior = evaluated_prior + cos_evaluations @ self.cos_coefficients + \
                                                sin_evaluations @ self.sin_coefficients

        # TODO: Type conversion in cases when only float.

        return evaluated_prior


    def kernel(self, point_1, point_2):
        """
        **Parameters**
        """
        # Check that point_1 and point_2 have the same dimension
        if type(point_1) == float or type(point_1) == int:
            number_of_points_1 = 1
        else:
            number_of_points_1 = point_1.size

        if type(point_2) == float or type(point_2) == int:
            number_of_points_2 = 1
        else:
            number_of_points_2 = point_2.size

        if number_of_points_1 != number_of_points_2:
            raise Exception(f"Arguments don't have same dimension.")
        else:
            number_of_points = number_of_points_1

        # Initialize by evaluating first eigenvalue x squared constant basis function
        kernel_evaluation = self.eigenvalues[0] * np.ones(number_of_points) / (2 * np.pi)  

        # Sort the eigenvalues per the cos and sin basis functions
        if self.half_number_of_eigenvalues != 0:
            reduced_eigenvalues = np.delete(self.eigenvalues, 0)
            cos_eigenvalues     = reduced_eigenvalues[0::2]
            sin_eigenvalues     = reduced_eigenvalues[1::2]

            cos_evaluation_1 = np.zeros((number_of_points, self.half_number_of_eigenvalues))
            cos_evaluation_2 = np.zeros((number_of_points, self.half_number_of_eigenvalues))
            sin_evaluation_1 = np.zeros((number_of_points, self.half_number_of_eigenvalues))
            sin_evaluation_2 = np.zeros((number_of_points, self.half_number_of_eigenvalues))

            for index in range(self.half_number_of_eigenvalues):
                cos_evaluation_1[:, index] = np.cos((index + 1) * point_1) / np.sqrt(np.pi)
                cos_evaluation_2[:, index] = np.cos((index + 1) * point_2) / np.sqrt(np.pi)
                sin_evaluation_1[:, index] = np.sin((index + 1) * point_1) / np.sqrt(np.pi)
                sin_evaluation_2[:, index] = np.sin((index + 1) * point_2) / np.sqrt(np.pi)

            # Add the cos and sin parts of the kernel
            kernel_evaluation = kernel_evaluation + \
                                cos_evaluation_1 * cos_evaluation_2 @ cos_eigenvalues + \
                                sin_evaluation_1 * sin_evaluation_2 @ cos_eigenvalues

        return kernel_evaluation










































