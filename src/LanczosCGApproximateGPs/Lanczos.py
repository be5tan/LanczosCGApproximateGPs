import numpy as np

class Lanczos():
    """
    Lanczos algorithm

    Parameters
    ----------
    matrix: array
        nxn-dim matrix

    starting_vector: array
        n-dim vector with unit Euclidean norm.

    Attributes
    ----------
    onb_matrix: array
        nxm-dim matrix of orthonormal basis vectors of the Krylov space, where m = number_of_eigenpairs


    """
    def __init__(self, matrix, starting_vector, number_of_eigenpairs):
        # User input
        self.matrix               = matrix
        self.starting_vector      = starting_vector
        self.number_of_eigenpairs = number_of_eigenpairs
        
        # Main and sub-diagonals in standard notation, see Saad 2011
        # Note that the algorithm has to iterate to beta_m+1
        self.alpha = np.zeros(self.number_of_eigenpairs)
        self.beta  = np.zeros(self.number_of_eigenpairs + 1)

        # ONB matrix and tridiagonal matrix
        self.onb_matrix         = np.zeros((np.shape(self.matrix)[0],  self.number_of_eigenpairs + 1))
        self.tridiagonal_matrix = np.zeros((self.number_of_eigenpairs, self.number_of_eigenpairs))

        # Eigenquantities
        self.eigenvectors = np.zeros((np.shape(self.matrix)[0],  self.number_of_eigenpairs))
        self.eigenvalues  = np.zeros(np.shape(self.matrix)[0])

        # Check quantities
        self.__did_run = False

    def run(self):
        """
        Runs the main algorithm
        """
        # Get initial quantities
        old_basis_vector      = np.zeros(np.shape(self.matrix)[0]) 
        basis_vector          = self.starting_vector
        self.onb_matrix[:, 0] = basis_vector  

        # main loop
        for index in range(self.number_of_eigenpairs):
            # Initialize new basis vector and first orthogonolization
            new_basis_vector = self.matrix @ basis_vector - self.beta[index] * old_basis_vector

            # Second orthogonolization step
            self.alpha[index] = np.dot(new_basis_vector, basis_vector)
            new_basis_vector = new_basis_vector - self.alpha[index] * basis_vector

            # Normalization step
            self.beta[index + 1] = np.sqrt(np.sum(new_basis_vector**2))
            new_basis_vector    = new_basis_vector / self.beta[index + 1]

            # Collect and update quantities
            self.onb_matrix[:, index + 1] = new_basis_vector
            old_basis_vector             = basis_vector
            basis_vector                 = new_basis_vector

        # Eliminate one superfluous column of the onb_matrix
        self.onb_matrix         = self.onb_matrix[:, 0:self.number_of_eigenpairs]

        # Fill the tridiagonal matrix
        self.tridiagonal_matrix = np.diag(self.beta[1:(self.number_of_eigenpairs)], -1) + \
                                  np.diag(self.alpha, 0) + \
                                  np.diag(self.beta[1:(self.number_of_eigenpairs)], 1)

        self.__did_run          = True

    # def get_eigenvectors():

    def get_eigenquantities(self):
        """Compute the Lanczos approximate eigenvectors"""
        # Lower dimensional SVD of the tridiagonal matrix
        U, S, Vh          = np.linalg.svd(self.tridiagonal_matrix, hermitian = True)

        self.eigenvectors = self.onb_matrix @ U
        self.eigenvalues  = S







