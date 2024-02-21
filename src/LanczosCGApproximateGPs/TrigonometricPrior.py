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
            self.cos_coefficients    = reduced_sqrt_eigenvalues[0::2] # * \
                                       # np.random.normal(0, 1, self.half_number_of_eigenvalues)
            self.sin_coefficients    = reduced_sqrt_eigenvalues[1::2] # * \
                                       # np.random.normal(0, 1, self.half_number_of_eigenvalues)

    def evaluate(self, points):
        """
        **Parameters**

        *point*: ``float or array``. Points from [-pi, pi] at which to evaluate the prior.
        """
        if type(points) == float or type(points) == int:
            number_of_points  = 1
        else:
            number_of_points  = points.shape[0]

        const_evaluations = np.ones(number_of_points) / np.sqrt(2 * np.pi)  
        evaluated_prior   = const_evaluations * self.const_coefficient

        if self.half_number_of_eigenvalues != 0:
            cos_evaluations = np.zeros((number_of_points, self.half_number_of_eigenvalues))
            sin_evaluations = np.zeros((number_of_points, self.half_number_of_eigenvalues))

            for index in range(self.half_number_of_eigenvalues):
                cos_evaluations[:, index] = np.cos((index + 1) * points)
                sin_evaluations[:, index] = np.sin((index + 1) * points)

            evaluated_prior = evaluated_prior + cos_evaluations @ self.cos_coefficients + \
                                                sin_evaluations @ self.sin_coefficients

        # TODO: adjust the cos evaluations so that the basis is normed to one. 
        return evaluated_prior
