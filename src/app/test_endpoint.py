#!/usr/bin/env python3
"""
Code for unit testing.
"""
# Standard Libraries
from io import BytesIO
from os import environ, makedirs
from os.path import join

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from numpy import frombuffer, uint8

# Local Files
from app.endpoints import app

# This is for testing; only using docstrings if the naming is not descriptive enough.
#pylint: disable=missing-function-docstring

TEST_FILES_DIR = environ.get("TEST_FILES_DIR")
INPUT_DIR = join(TEST_FILES_DIR, "input")
OUTPUT_DIR = join(TEST_FILES_DIR, "output", "test_endpoint")
makedirs(OUTPUT_DIR, exist_ok=True)

def test_endpoint_detect_handgun():
    with open(join(INPUT_DIR, "test_00.jpg"), "rb") as image_file:
        test_image = image_file.read()

    with app.test_client() as client:
        data = {}
        data["file"] = (BytesIO(test_image), "image.jpg")
        response = client.post("/detect_handguns", data=data, content_type="multipart/form-data")
        assert response.status_code == 200

        image_bytes = response.get_data()
        numpy_array = frombuffer(image_bytes, uint8)
        result_image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        cv2.imwrite(join(OUTPUT_DIR, "test_00.jpg"), result_image)

        # Test that having no file fails.
        response = client.post("/detect_handguns")
        assert response.status_code == 400
