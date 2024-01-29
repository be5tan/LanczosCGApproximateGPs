import unittest
import numpy as np
from LanczosCGApproximateGPs import Iter_GP

class Test_Iter_GP(unittest.TestCase):
    """Tests for the Iter_GP class."""

    def setUp(self):
        self.sample_size = 5
        self.max_iter    = self.sample_size
        self.noise_level = 1   

        self.X = np.array([np.linspace(0, 1, self.sample_size)])
        self.X = self.X.transpose()
        self.Y = self.X[:, 0]**2     

        self.actions = []
        index = 0
        for iter in range(0, self.max_iter):
            action = np.zeros(self.sample_size)
            action[index] = 1
            self.actions.append(action)
            index = index + 1

        def brownian_kernel(x, y):
            z = np.minimum(x, y)
            return z

        self.kernel = brownian_kernel


    def test_inversion_for_small_dimensions(self):
        self.algo = Iter_GP(self.X, self.Y, self.noise_level, self.actions, self.kernel)
        self.algo.iter_forward(self.sample_size)
        iterative_inverse = self.algo.approx_augmented_inverse

        augmented_kernel_matrix = self.algo.augmented_kernel_matrix
        inverse                 = np.linalg.inv(augmented_kernel_matrix)

        decimal_place = 5
        message       = "Inverses do not coincide"
        for i in range(self.sample_size):
            for j in range(self.sample_size):
                self.assertAlmostEqual(iterative_inverse[i, j], inverse[i, j], decimal_place, message)

if __name__ == '__main__': 
    unittest.main() 
