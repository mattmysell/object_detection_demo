#!/usr/bin/env python3
"""
Code for comparing speed of detect and detect_batch.
"""
# Standard Libraries

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.

# Local Files
from object_detection.detect import detect
from object_detection.detect_batch import detect_batch
from object_detection.metadata import get_model_metadata
from object_detection.utils import print_statistics

if __name__ == "__main__":
    model_metadata = get_model_metadata("handguns")
    result_images = []
    inference_milliseconds = []
    for i in range(6):
        in_path = f"./object_detection/test/test_{str(i).zfill(2)}.jpg"
        result_image, inference_second = detect(in_path, model_metadata)
        result_images.append(result_image)
        inference_milliseconds.append(inference_second*1000)
    for i, result_image in enumerate(result_images):
        out_path = f"./object_detection/output/test_{str(i).zfill(2)}_detect.jpg"
        cv2.imwrite(out_path, result_image)
    print_statistics(inference_milliseconds, len(result_images))

    model_metadata = get_model_metadata("handguns")
    input_images = [f"./object_detection/test/test_{str(i).zfill(2)}.jpg" for i in range(6)]
    inference_seconds = detect_batch(input_images, model_metadata, 3)
    print_statistics([inference_seconds*1000], len(input_images))
