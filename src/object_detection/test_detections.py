#!/usr/bin/env python3
"""
Code for testing the functions and methods in detections.
"""
# Standard Libraries

# Installed Libraries
from numpy import array

# Local Files
from detections import is_type, is_type_list

# This is for testing, only using docstrings if the naming is not descriptive enough.
#pylint: disable=missing-function-docstring

class CannotStr: #pylint: disable=too-few-public-methods
    """
    Class with only purpose is to be unable to convert to str.
    """
    def __str__(self):
        pass

## Testing is_type... ##

def test_is_type_with_int():
    assert is_type(10, int)
    assert is_type("10", int)
    assert not is_type("abc", int)

def test_is_type_with_none():
    assert not is_type(None, int)

def test_is_type_with_str():
    assert is_type("example", str)
    assert is_type(123, str)
    assert not is_type(CannotStr(), str)

def test_is_type_with_float():
    assert is_type(10, float)
    assert is_type("10.5", float)
    assert is_type("10", float)
    assert not is_type("abc", float)

def test_is_type_with_exact_types():
    assert not is_type(10.5, int, exact_type=True)
    assert not is_type("10.5", float, exact_type=True)

## Testing is_type_list... ##

def test_is_type_list_with_correct_type():
    assert is_type_list([10, 20, 30], int)
    assert is_type_list((10, 20, 30), int)
    assert is_type_list(array([10, 20, 30]), int)

def test_is_type_list_with_mixed_types():
    assert is_type_list([10, "20", 30], int)
    assert not is_type_list([10, "20", 30], int, exact_type=True)

def test_is_type_list_with_empty_list():
    assert is_type_list([], int)

def test_is_type_list_with_none():
    assert not is_type_list(None, int)

def test_is_type_list_with_specific_length():
    assert is_type_list([10, 20], int, list_length=2)
    assert is_type_list([10, 20], int, list_length=(2, 3))
    assert not is_type_list([10], int, list_length=2)
    assert not is_type_list([10], int, list_length=(2, 3))
