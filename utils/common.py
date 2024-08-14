'''
Placeholder for common functions used by PIPT and POPT
'''

import numpy as np
from scipy.linalg import solve
def vectorize(value, keys, filter=None, return_map=False):
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
    return_map: bool, optional
        If True, returns a dictionary with keys equal to the original input and values as a list of integers
        that give the index of the original value in the concatenated array.

    Returns
    -------
    np.ndarray or tuple
        A concatenated numpy array of values if `value` is not None. If `value` is None, returns None.
        If `return_map` is True, returns a tuple of the concatenated array and the index map dictionary.

    """

    if value is None:
        return None
    elif type(value) is dict:
        # Retrieve the arrays corresponding to the  keys
        if filter is None:
            arrays_to_concatenate = [value[key] for key in keys]
        elif type(filter) is dict:
            arrays_to_concatenate = [value[key] for key in keys if filter[key]]
        else:
            raise TypeError("Filter must be either None or a dictionary.")

        try:
            concatenated_array = np.concatenate(arrays_to_concatenate)
        except ValueError as e:
            raise ValueError("Elements to be vectorized have unequal ensemble size!") from e

        if return_map:
            index_map = {}
            start_idx = 0
            for key in keys:
                if filter is None or filter[key]:
                    end_idx = start_idx + len(value[key])
                    index_map[key] = list(range(start_idx, end_idx))
                    start_idx = end_idx
            return concatenated_array, index_map

        return concatenated_array
    elif type(value) is list:
        if filter is None:
            tuple_to_concatenate = tuple(val[key] for val in value if val is not None for key in keys)
        elif type(filter) is list:
            tuple_to_concatenate = tuple(val[key] for ind, val in enumerate(value) if val is not None for key in keys if filter[ind][key])
        else:
            raise TypeError("Filter must be either None or a list of dictionaries.")

        try:
            concatenated_array = np.concatenate(tuple_to_concatenate)
        except ValueError as e:
            raise ValueError("Elements to be vectorized have unequal ensemble size!") from e
        if return_map:
            index_map = {}
            start_idx = 0
            for ind, val in enumerate(value):
                if val is not None:
                    for key in keys:
                        if filter is None or filter[ind][key]:
                            end_idx = start_idx + len(val[key])
                            index_map.setdefault(ind, {})[key] = list(range(start_idx, end_idx))
                            start_idx = end_idx
            return concatenated_array, index_map

        return concatenated_array
    else:
        raise TypeError(f"Input value {type(value)} is not supported. Must be either a dictionary or a list of dictionaries.")

def unpack_vector(value, keys, index_map, filter=None, out_type=dict):
    """
    Rebuild the original input structure from the concatenated array and index map.

    Parameters
    ----------
    value: np.ndarray
        The concatenated array of values.
    keys: list
        List of keys.
    index_map: dict
        Dictionary with keys equal to the original input and values as a list of integers
        that give the index of the original value in the concatenated array.
    filter: None, dict or list, optional
        - If None: No filter is applied (the default)
        - If dict: Each key contains a boolean value
        - If list: Each list element contains a dictionary with boolean values.
    out_type: type, optional
        The type of the output structure (default is dict).

    Returns
    -------
    dict or list
        The rebuilt original input structure.
    """
    if out_type not in [dict, list]:
        raise TypeError("out_type must be either dict or list.")
    if out_type is dict:
        rebuilt_value = {}
        for key in keys:
            if key in index_map:
                indices = index_map[key]
                if len(value.shape) >1:
                    rebuilt_value[key] = value[indices,:]
                else:
                    rebuilt_value[key] = value[indices]
                if filter and key in filter and not filter[key]:
                    del rebuilt_value[key]
    elif out_type is list:
        rebuilt_value = []
        for ind, key_map in index_map.items():
            item = {}
            for key in keys:
                if key in key_map:
                    indices = key_map[key]
                    if len(value.shape) >1:
                        item[key] = value[indices,:]
                    else:
                        item[key] = value[indices]
                    if filter and filter[ind][key] is False:
                        del item[key]
            rebuilt_value.append(item)

    return rebuilt_value


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
        if len(b.shape) == 1:
            x = b / A
        else:
            x = b / A[:, np.newaxis]
    else:
        # Solve the linear system if A is a 2D array
        x = solve(A, b)

    return x