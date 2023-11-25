#!/usr/bin/env python3
"""
Code for running object detection on an image, more suitable for CPUs.
"""

# Standard Libraries
from os import environ
from time import perf_counter
from typing import List, Tuple, Union

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from numpy.typing import NDArray
from ovmsclient import make_grpc_client

# Local Files
from detections import Detections
from utils import print_statistics

# Create connection to the model server
HOST = environ.get("INFERENCE_HOST", "localhost")
PORT = environ.get("INFERENCE_PORTT", "9000")
CLIENT = make_grpc_client(f"{HOST}:{PORT}")

def detect(image: Union[str, NDArray], model_name: str, model_classes: List[str],
           model_shape: List[int]) -> Tuple[NDArray, float]:
    """
    class_names => list of class names, can also be a tuple or numpy array.
    input_array => output from an object detection predicition, can be in various formats.
    input_type => str for the type of input_array format.
    model_shape => pixel height and width of the models output shape.

    Detect the objects in an image and return an image with the results and the inference time in seconds.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    blobs = cv2.dnn.blobFromImage(image, 1/255, model_shape, [0,0,0], 1, crop=False)

    inference_start = perf_counter()
    output = CLIENT.predict(inputs={"images": blobs}, model_name=model_name)
    output = cv2.transpose(output[0])

    detections = Detections(model_classes, output, model_shape=model_shape)
    detections.apply_non_max_suppression()
    # Include apply non max suppression in inference as it is part of the process.
    inference_end = perf_counter()

    image = detections.draw(image)
    return image, inference_end - inference_start

if __name__ == "__main__":
    result_images = []
    inference_milliseconds = []

    for i in range(6):
        in_path = f"./images/test_{str(i).zfill(2)}.jpg"
        result_image, inference_second = detect(in_path, "handguns", ["handgun"], (480, 480))
        result_images.append(result_image)
        inference_milliseconds.append(inference_second*1000)

    for i, result_image in enumerate(result_images):
        out_path = f"./output/test_{str(i).zfill(2)}_detect.jpg"
        cv2.imwrite(out_path, result_image)

    print_statistics(inference_milliseconds, len(result_images))
