import torch

"""
This module provides utility functions for arrays 
that are relevant to the project.
"""

def triangular_power_matrix(side_length: int, base: float) -> torch.Tensor:

    """
    Returns a lower triangular square matrix whose entries are powers of a given base.

    Arguments:

        `int` side_length: the side length of the matrix.

        `float` base: the base to exponentiate.

    Examples:

        Input: `side_length` = 4, `base` = 2

        Output:

        `[1 0 0 0]`

        `[2 1 0 0]`

        `[4 2 1 0]`

        `[8 4 2 1]`

        Input: `side_length` = 6, `base` = 5

        Output:

        `[1 0 0 0 0 0]`

        `[5 1 0 0 0 0]`

        `[25 5 1 0 0 0]`

        `[125 25 5 1 0 0]`

        `[625 125 25 5 1 0]`

        `[3125 625 125 25 5 1]`
    """
        
    # makes a matrix where each row is a copy of the arange
    arange_matrix: np.array = np.arange(side_length, 0, -1) + np.zeros((side_length, side_length))

    powers: np.array = np.maximum(np.zeros((side_length, side_length)), arange_matrix - arange_matrix.T)

    # used to mask elements above the main diagonal to 0
    return np.tril(base ** powers).astype(type(base))