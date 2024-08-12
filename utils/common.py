'''
Placeholder for common functions used by PIPT and POPT
'''

import numpy as np
from scipy.linalg import solve
def vectorize(value, keys, filter=None):
    """
    Vectorize the input.

    Parameters
    ----------
    value: dict or list
        The data to vectorize.
        - If dict: Each key contains a value as numpy array.
        - If list: Each list value contains a dictionary where the dictionaries values are numpy arrays.
    keys: list
        List of keys.
    filter: None, dict or list
        - If None: No filter is applied (the default)
        - If dict: Each key contains a boolean value
        - If list: Each list element contains a dictionary with boolean values.

    Returns
    -------
    np.ndarray
        A concatenated numpy array of values if `value` is not None. If `value` is None, returns None.

    """

    # Sort the list of keys to ensure consistent order
    sorted_keys = sorted(keys)

    if value is None:
        return None
    elif type(value) is dict:
        # Retrieve the arrays corresponding to the sorted keys
        if filter is None:
            arrays_to_concatenate = [value[key] for key in sorted_keys]
        elif type(filter) is dict:
            arrays_to_concatenate = [value[key] for key in sorted_keys if filter[key]]
        else:
            raise TypeError("Filter must be either None or a dictionary.")
        # Concatenate the arrays
        return np.concatenate(arrays_to_concatenate)
    elif type(value) is list:
        if filter is None:
            tuple_to_concatenate = tuple(val[key] for val in value if val is not None for key in keys)
        elif type(filter) is list:
            tuple_to_concatenate = tuple(val[key] for ind,val in enumerate(value) if val is not None for key in keys if
                                         filter[ind][key])
        else:
            raise TypeError("Filter must be either None or a list of dictionaries.")
        #Concatenate the tuples
        return np.concatenate(tuple_to_concatenate)
    # give error if any other type is used
    else:
        raise TypeError(f"Input value {type(value)} is not supported. Must be either a dictionary or a list of "
                        f"dictionaries.")


def solve_linear(A, b):
    """
    Solves a linear system of equations or performs element-wise division.

    Parameters
    ----------
    A : numpy.ndarray
        The coefficient matrix. If `A` is a 1D array, element-wise division is performed.
        If `A` is a 2D array, it is treated as a matrix for solving the linear system.
    b : numpy.ndarray
        The right-hand side array or matrix.

    Returns
    -------
    numpy.ndarray
        The solution to the linear system or the result of the element-wise division.
    """
    if len(A.shape) == 1:
        # Perform element-wise division if A is a 1D array
        x = np.dot(np.expand_dims(A ** (-1), axis=1), np.ones((1, b.shape[1]))) * b
    else:
        # Solve the linear system if A is a 2D array
        x = solve(A, b)

    return x