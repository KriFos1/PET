import numpy as np
from utils.common import vectorize, unpack_vector, solve_linear
import pytest
def test_vectorize_dict_no_filter():
    value = {'a': np.array([1, 2]), 'b': np.array([3, 4])}
    keys = ['a', 'b']
    result = vectorize(value, keys)
    expected = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(result, expected)
def test_vectorize_dict_no_filter_matrix():
    value = {
        'a': np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
        'b': np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    }
    keys = ['a', 'b']
    result = vectorize(value, keys)
    expected = np.array([
        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]
    ])
    np.testing.assert_array_equal(result, expected)
def test_vectorize_dict_with_filter():
    value = {'a': np.array([1, 2]), 'b': np.array([3, 4])}
    keys = ['a', 'b']
    filter = {'a': True, 'b': False}
    result = vectorize(value, keys, filter)
    expected = np.array([1, 2])
    np.testing.assert_array_equal(result, expected)

def test_vectorize_list_no_filter():
    value = [{'a': np.array([1, 2]), 'b': np.array([3, 4])}, {'a': np.array([5, 6]), 'b': np.array([7, 8])}]
    keys = ['a', 'b']
    result = vectorize(value, keys)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(result, expected)
def test_vectorize_list_no_filter_matrix():
    value = [
        {'a': np.array([[1, 2, 3], [4, 5, 6]]), 'b': np.array([[7, 8, 9], [10, 11, 12]])},
        {'a': np.array([[13, 14, 15], [16, 17, 18]]), 'b': np.array([[19, 20, 21], [22, 23, 24]])}
    ]
    keys = ['a', 'b']
    result = vectorize(value, keys)
    expected = np.array([
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
        [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]
    ])
    np.testing.assert_array_equal(result, expected)
def test_vectorize_list_with_filter():
    value = [{'a': np.array([1, 2]), 'b': np.array([3, 4])}, {'a': np.array([5, 6]), 'b': np.array([7, 8])}]
    keys = ['a', 'b']
    filter = [{'a': True, 'b': False}, {'a': False, 'b': True}]
    result = vectorize(value, keys, filter)
    expected = np.array([1, 2, 7, 8])
    np.testing.assert_array_equal(result, expected)

def test_vectorize_return_map():
    value = {'a': np.array([1, 2]), 'b': np.array([3, 4])}
    keys = ['a', 'b']
    result, index_map = vectorize(value, keys, return_map=True)
    expected = np.array([1, 2, 3, 4])
    expected_map = {'a': [0, 1], 'b': [2, 3]}
    np.testing.assert_array_equal(result, expected)
    assert index_map == expected_map

def test_unpack_vector_dict():
    value = np.array([1, 2, 3, 4])
    keys = ['a', 'b']
    index_map = {'a': [0, 1], 'b': [2, 3]}
    result = unpack_vector(value, keys, index_map)
    expected = {'a': np.array([1, 2]), 'b': np.array([3, 4])}
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
def test_unpack_vector_dict_matrix():
    value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    keys = ['a', 'b']
    index_map = {'a': [0, 1], 'b': [2, 3]}
    result = unpack_vector(value, keys, index_map)
    expected = {
        'a': np.array([[1, 2, 3], [4, 5, 6]]),
        'b': np.array([[7, 8, 9], [10, 11, 12]])
    }
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
def test_unpack_vector_list():
    value = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    keys = ['a', 'b']
    index_map = {0: {'a': [0, 1], 'b': [2, 3]}, 1: {'a': [4, 5], 'b': [6, 7]}}
    result = unpack_vector(value, keys, index_map, out_type=list)
    expected = [{'a': np.array([1, 2]), 'b': np.array([3, 4])}, {'a': np.array([5, 6]), 'b': np.array([7, 8])}]
    for i, item in enumerate(expected):
        for key in item:
            np.testing.assert_array_equal(result[i][key], item[key])
def test_unpack_vector_list_matrix():
    value = np.array([
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
        [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]
    ])
    keys = ['a', 'b']
    index_map = {0: {'a': [0, 1], 'b': [2, 3]}, 1: {'a': [4, 5], 'b': [6, 7]}}
    result = unpack_vector(value, keys, index_map, out_type=list)
    expected = [
        {'a': np.array([[1, 2, 3], [4, 5, 6]]), 'b': np.array([[7, 8, 9], [10, 11, 12]])},
        {'a': np.array([[13, 14, 15], [16, 17, 18]]), 'b': np.array([[19, 20, 21], [22, 23, 24]])}
    ]
    for i, item in enumerate(expected):
        for key in item:
            np.testing.assert_array_equal(result[i][key], item[key])
def test_vectorize_invalid_input():
    with pytest.raises(TypeError):
        vectorize(123, ['a', 'b'])

def test_unpack_vector_invalid_out_type():
    with pytest.raises(TypeError):
        unpack_vector(np.array([1, 2, 3, 4]), ['a', 'b'], {'a': [0, 1], 'b': [2, 3]}, out_type=str)

def test_solve_linear_1d_array():
    A = np.array([2, 4])
    b = np.array([[1, 2], [3, 4]])
    result = solve_linear(A, b)
    expected = np.array([[0.5, 1], [0.75, 1]])
    np.testing.assert_array_almost_equal(result, expected)

def test_solve_linear_2d_array():
    A = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])
    result = solve_linear(A, b)
    expected = np.array([2, 3])
    np.testing.assert_array_almost_equal(result, expected)

def test_solve_linear_invalid_input():
    A = np.array([1, 2, 3])
    b = np.array([4, 5])
    with pytest.raises(ValueError):
        solve_linear(A, b)