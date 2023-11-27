#!/usr/bin/env python3
"""
Code for testing the functions and methods in detections.
"""
# Standard Libraries
from json import load
from os.path import dirname, join, realpath

# Installed Libraries
from numpy import array, ones
from pytest import fixture, raises

# Local Files
from object_detection.detections import is_type, is_type_list, Detections

# This is for testing; only using docstrings if the naming is not descriptive enough, ignore access to protected, and
# ignore redefining outer name for fixtures.
#pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

THIS_DIR = dirname(realpath(__file__))

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

## Testing Detections Class - __init__ ##

def test_init_with_valid_inputs():
    params = {"class_names": ["test"], "input_array": ones((10, 5)), "input_type": "xcycwhps",
              "model_shape": [640., 480.]}
    detections = Detections(**params)
    assert detections.class_names == params["class_names"]
    assert detections.shape == (480, 640)

    params = {"class_names": ["test"], "input_array": ones((10, 5)), "model_shape": [640., 480.]}
    detections = Detections(**params)
    assert detections.class_names == params["class_names"]
    assert detections.shape == (480, 640)

def test_init_with_invalid_inputs():
    params = {"class_names": "test", "input_array": ones((10, 5)), "input_type": "xcycwhps", "model_shape": [640, 480]}
    with raises(ValueError) as exc:
        _detections = Detections(**params)
        assert str(exc) == "Class names is not a list of str"

    params = {"class_names": ["test"], "input_array": ones((10, 5)), "input_type": "test", "model_shape": [640, 480]}
    with raises(NotImplementedError):
        _detections = Detections(**params)
        assert str(exc) == "Input type is not implemented"

## Testing Detections Class - _import_type_xcycwhps ##

@fixture()
def detections_base():
    params = {"class_names": ["test"], "input_array": ones((10, 5)), "input_type": "xcycwhps",
              "model_shape": [640., 480.]}
    return Detections(**params)

@fixture()
def detections_real():
    with open(join(THIS_DIR, "test", "test_xcycwhps_input.json"), "r") as input_json:
        input_array = load(input_json)
    params = {"class_names": ["handguns"], "input_array": input_array, "input_type": "xcycwhps",
              "model_shape": [480, 480]}
    return Detections(**params)

def test_import_type_xcycwhps_with_invalid_inputs(detections_base):
    params = {"input_array": ones((10, 4)), "model_shape": [480]}
    with raises(ValueError) as exc:
        detections_base._import_type_xcycwhps(**params)
        assert str(exc) == "Invalid model_shape provided for \"xcycwh\" type, expected [int, int]"

    params = {"input_array": 4.0, "model_shape": [480, 480]}
    with raises(ValueError) as exc:
        detections_base._import_type_xcycwhps(**params)
        assert str(exc) == "Invalid input_array provided for \"xcycwhps\" type, expected a 2D list of floats"

    params = {"input_array": [4.0], "model_shape": [480, 480]}
    with raises(ValueError) as exc:
        detections_base._import_type_xcycwhps(**params)
        assert str(exc) == "Invalid input_array provided for \"xcycwhps\" type, expected a 2D list of floats"

    params = {"input_array": [["a"]], "model_shape": [480, 480]}
    with raises(ValueError) as exc:
        detections_base._import_type_xcycwhps(**params)
        assert str(exc) == "Invalid input_array provided for \"xcycwhps\" type, expected a 2D list of floats"

    params = {"input_array": ones((10, 4)), "model_shape": [480, 480]}
    with raises(ValueError) as exc:
        detections_base._import_type_xcycwhps(**params)
        assert str(exc) == "Invalid input_array provided for \"xcycwhps\" type, expected shape (*, >=5)"

def test_import_type_xcycwhps_with_valid_inputs(detections_real):
    with open(join(THIS_DIR, "test", "test_xcycwhps_expected.json"), "r") as expected_json:
        expected_output = load(expected_json)
    assert [detections_real.boxes[i] == box for i, box in enumerate(expected_output["boxes"])]
    assert [detections_real.confidences[i] == box for i, box in enumerate(expected_output["confidences"])]
    assert [detections_real.classes[i] == box for i, box in enumerate(expected_output["classes"])]

## Testing Detections Class - apply_non_max_suppression ##

def test_apply_non_max_suppression_with_valid_inputs(detections_real):
    with open(join(THIS_DIR, "test", "test_xcycwhps_nms_expected.json"), "r") as expected_json:
        expected_output = load(expected_json)
    detections_real.apply_non_max_suppression()
    assert [detections_real.boxes[i] == box for i, box in enumerate(expected_output["boxes"])]
    assert [detections_real.confidences[i] == box for i, box in enumerate(expected_output["confidences"])]
    assert [detections_real.classes[i] == box for i, box in enumerate(expected_output["classes"])]
