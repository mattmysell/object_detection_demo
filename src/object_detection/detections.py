#!/usr/bin/env python3
"""
Code for a detections class to describe and transform the results of object detection.
"""
# Standard Libraries
from typing import Callable, List, Union

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from numpy import ndarray
from numpy.typing import NDArray

# Local Files

CONFIDENCE_THRESHOLD = 0.25 # Low confidence threshold as we want to lean towards the side of caution.
OVERLAP_THRESHOLD = 0.4

def is_type(value: any, allowed_types: Union[any, List[any], None], exact_type: bool=False) -> bool:
    """
    value => any variable.
    allowed_types => function to convert type.

    Check if the type is expected.
    """
    if exact_type:
        return isinstance(value, allowed_types)

    try:
        allowed_types(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def is_type_list(value: any, type_function: Callable, exact_type: bool=False,
                 list_length: Union[int, List[int], None]=None) -> bool:
    """
    value => any variable.
    type_function => function to convert type, for the variables in the list.
    list_length => optional int or list of int for the expected lengths of a list.

    Check if the type is a list of a specific type.
    """
    if not isinstance(value, (list, tuple, ndarray)):
        return False
    if isinstance(list_length, int) and (not len(value) == list_length):
        return False
    if isinstance(list_length, (list, tuple)) and not any(len(value) == length for length in list_length):
        return False
    if any(not is_type(value, type_function, exact_type) for value in value):
        return False
    return True

class Detections():
    """
    Detections class.
    """
    def __init__(self, class_names: List[str], input_array: any, input_type: str="xcycwhps",
                 model_shape: List[int]=None):
        """
        class_names => list of class names, can also be a tuple or numpy array.
        input_array => output from an object detection predicition, can be in various formats.
        input_type => str for the type of input_array format.
        model_shape => pixel height and width of the models output shape.

        In this Detections class we want to use percentage of the image size, as a fraction, so that values are easily
        scaled to various image sizes.
        """
        if not is_type_list(class_names, str, exact_type=True):
            raise ValueError("Class name is not a list of str")

        self.class_names = class_names
        self.shape = (1, 1) # (width, height)
        self.boxes = [] # list of (center x, center y, w, h) => (float, float, float, float)
        self.confidences = [] # list of float
        self.classes = [] # list of int
        self.nms_indexs = None # list of int | None, for the final indexes after applying non max suppression.

        if input_type == "xcycwhps":
            self._import_type_xcycwhps(input_array, model_shape)
        else:
            raise NotImplementedError("Coordinate type is not implemented")

    def _import_type_xcycwhps(self, input_array: List[List[float]], model_shape: List[int]):
        """
        input_array => a 2D list, tuple or ndarray of floats.
        model_shape => pixel height and width of the models output shape.
        """
        if not is_type_list(input_array, list):
            raise ValueError("Invalid input_array provided for \"xcycwhps\" type, expected a 2D list of floats")

        if len(input_array) == 0:
            # No detections so do nothing.
            return

        if not is_type_list(input_array[0], float):
            raise ValueError("Invalid input_array provided for \"xcycwhps\" type, expected a 2D list of floats")

        if not is_type_list(model_shape, int, list_length=(2, 3)):
            raise ValueError("Invalid model_shape provided for \"xcycwh\" type, expected [int, int]")

        self.shape = (int(model_shape[1]), int(model_shape[0])) # flip the input shape values around.

        for row in input_array:
            # Each row is [center x, center y, width, height, probability class 0, probability class 1, ...]
            class_confidences = row[4:]
            (_min_confidence, max_confidence, _min_class_loc, (_x, max_class_index)) = cv2.minMaxLoc(class_confidences)
            if max_confidence >= CONFIDENCE_THRESHOLD:
                center_x = float(row[0])/self.shape[0]
                center_y = float(row[1])/self.shape[1]
                width = float(row[2])/self.shape[0]
                height = float(row[3])/self.shape[1]
                self.boxes.append((center_x, center_y, width, height))
                self.confidences.append(max_confidence)
                self.classes.append(max_class_index)

    def apply_non_max_suppression(self):
        """
        Apply the non max suppresion method to the detection results.
        """
        self.nms_indexs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, CONFIDENCE_THRESHOLD, OVERLAP_THRESHOLD)

    def draw(self, image: NDArray, include_confidence: bool=True, color: List[int]=None) -> NDArray:
        """
        color => list of int, in BGR format.

        Draw bounding boxes around the detections in an image, returning the resulting image.
        """
        color = color if color is not None else (0, 0, 255)
        if not is_type_list(color, int, exact_type=True, list_length=3):
            raise ValueError("Invalid color provided, expected a list of 3 ints (b, g, r)")

        [height, width, _] = image.shape
        indexes = range(len(self.boxes)) if self.nms_indexs is None else self.nms_indexs
        for i in indexes:
            [x, y, w, h] = self.boxes[i]
            p1 = (round(width*(x - w/2)), round(height*(y - h/2)))
            p2 = (round(width*(x + w/2)), round(height*(y + h/2)))
            image = cv2.rectangle(image, p1, p2, color, 2)

            if include_confidence:
                image = cv2.putText(image, f"{self.class_names[self.classes[i]]} {round(self.confidences[i], 2)}",
                    (p1[0] - 2, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image
