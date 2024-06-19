import unittest
import numpy as np
from LanczosCGApproximateGPs import TrigonometricPrior

class Test_TrigonometricPrior(unittest.TestCase):
    """Tests for the TrigonometricPrior class."""

    def test_constant_function(self):
        eigenvalues = np.array([1])
        self.prior  = TrigonometricPrior(eigenvalues)

        value1 = self.prior.evaluate(-2)
        value2 = self.prior.evaluate(2.3)

        message = "Prior with only one coefficient is not constant."
        self.assertEqual(value1, value2, message)

    # This test is only valid after commenting out the randomness in __init__
    def test_alternative_computation(self):
        eigenvalues = np.array([1, 2, 3])
        self.prior  = TrigonometricPrior(eigenvalues)
        print(self.prior.eigenvalues)

        # Manual computation
        x = np.random.uniform(size = 1, low = -np.pi, high = np.pi)
        evaluated_prior = 1 / np.sqrt(2 * np.pi) + np.sqrt(2) * np.cos(x) \
                                                 + np.sqrt(3) * np.sin(x)

        message = "Prior does not coincide with manual computation."
        self.assertEqual(evaluated_prior, self.prior.evaluate(x))

        eigenvalues = np.array([1, 2, 3, 4, 5])
        self.prior  = TrigonometricPrior(eigenvalues)
        print(self.prior.eigenvalues)

        # Manual computation
        x = np.random.uniform(size = 1, low = -np.pi, high = np.pi)
        evaluated_prior = 1 / np.sqrt(2 * np.pi) \
                        + np.sqrt(2) * np.cos(x)     + np.sqrt(3) * np.sin(x) \
                        + np.sqrt(4) * np.cos(2 * x) + np.sqrt(5) * np.sin(2 * x)

        message = "Prior does not coincide with manual computation."
        self.assertEqual(evaluated_prior, self.prior.evaluate(x))

if __name__ == '__main__': 
    unittest.main() 
