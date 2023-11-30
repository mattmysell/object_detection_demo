#!/usr/bin/env python3
"""
Code for comparing speed of detect and detect_batch.

This can be run on your PC or from inside the objection_detection_demo_app_local docker container.

Note, we have not optimized the docker container for using Tensorflow with GPU, so this is an unfair comparison but it
does provide an example of how you might go about writing the code for both CPU and GPU instances.
"""
# Standard Libraries
from os import environ, makedirs
from os.path import dirname, join, pardir, realpath

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.

# Local Files
from object_detection.detect import detect
from object_detection.detect_batch import detect_batch
from object_detection.metadata import get_model_metadata
from object_detection.utils import print_statistics

THIS_DIR = dirname(realpath(__file__))
TEST_FILES_DIR = environ.get("TEST_FILES_DIR", join(THIS_DIR, pardir, "test_files"))
INPUT_DIR = join(TEST_FILES_DIR, "input")
OUTPUT_DIR = join(TEST_FILES_DIR, "output", "try_detect")
makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    model_metadata = get_model_metadata("handguns")
    result_images = []
    inference_milliseconds = []

    for i in range(6):
        in_path = join(INPUT_DIR, f"test_{str(i).zfill(2)}.jpg")
        result_image, inference_second = detect(in_path, model_metadata)
        result_images.append(result_image)
        inference_milliseconds.append(inference_second*1000)
    for i, result_image in enumerate(result_images):
        out_path = join(OUTPUT_DIR, f"test_{str(i).zfill(2)}_detect.jpg")
        cv2.imwrite(out_path, result_image)
    print_statistics(inference_milliseconds, len(result_images))

    model_metadata = get_model_metadata("handguns")
    input_images = [join(INPUT_DIR, f"test_{str(i).zfill(2)}.jpg") for i in range(6)]
    result_images, inference_seconds = detect_batch(input_images, model_metadata, 3)
    for i, result_image in enumerate(result_images):
        out_path = join(OUTPUT_DIR, f"test_{str(i).zfill(2)}_detect_batch.jpg")
        cv2.imwrite(out_path, result_image)
    print_statistics([inference_seconds*1000], len(input_images))
