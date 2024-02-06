import numpy as np

def get_conjugate_gradient_actions(matrix, response, max_iteration):
    """
    Computes the search directions the a conjugate gradient algorithm applied to A v = y up to a given iteration number.

    Parameters
    ----------
    matrix
        nxd-dim matrix.

    response: array
        n-dim vector of observed data.
    """
    # Initializing quantities
    cg_actions   = []
    sample_size  = np.shape(response)[0]
    iter         = 0
    solution     = np.zeros(sample_size)
    cg_gradient  = -response
    cg_direction =  response

    for iter in range(max_iteration):
        # Collect action
        action = cg_direction
        cg_actions.append(action)

        # CG-step
        step_size       = np.sum(cg_gradient**2) / \
                          np.dot(cg_direction, matrix @ cg_direction)
        solution        = solution + step_size * cg_direction

        # CG-updates
        cg_gradient_new = cg_gradient + step_size * matrix @ cg_direction
        weight          = np.sum(cg_gradient_new**2) / np.sum(cg_gradient**2)
        cg_gradient     = cg_gradient_new
        cg_direction    = -cg_gradient + weight * cg_direction

    return cg_actions
