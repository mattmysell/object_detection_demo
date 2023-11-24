#!/usr/bin/env python3
"""
Code for running object detection on an image.
"""

# Standard Libraries
from typing import List, Union

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from numpy.typing import NDArray
from ovmsclient import make_grpc_client

# Local Files
from detections import Detections

# Create connection to the model server
CLIENT = make_grpc_client("localhost:9000")

def detect(image:Union[str, NDArray], model_name:str, model_classes:List[str], model_shape:List[int])->NDArray:
    """
    class_names => list of class names, can also be a tuple or numpy array.
    input_array => output from an object detection predicition, can be in various formats.
    input_type => str for the type of input_array format.
    model_shape => pixel height and width of the models output shape.

    Detect the objects in an image and return an image with the results.
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    blobs = cv2.dnn.blobFromImage(image, 1/255, model_shape, [0,0,0], 1, crop=False)
    output = CLIENT.predict(inputs={"images": blobs}, model_name=model_name)
    output = cv2.transpose(output[0])

    detections = Detections(model_classes, output, model_shape=model_shape)
    detections.apply_non_max_suppression()
    image = detections.draw(image)

    return image

if __name__ == "__main__":
    cv2.imwrite("./output/test_image_00_detect.jpg",
                detect("./images/test_image_00.jpg", "handguns", ["handgun"], (480, 480)))
