#!/usr/bin/env python3
"""
Code for unit testing.
"""
# Standard Libraries
from io import BytesIO
from os.path import dirname, join, realpath

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from numpy import frombuffer, uint8

# Local Files
from app.endpoints import app

THIS_DIR = dirname(realpath(__file__))

def test_endpoint_detect_handgun():
    """
    test_endpoint_detect_handgun() --> null
    """
    with open(join(THIS_DIR, "test", "test_00.jpg"), "rb") as image_file:
        test_image = image_file.read()

    with app.test_client() as client:
        data = {}
        data["file"] = (BytesIO(test_image), "image.jpg")
        response = client.post("/detect_handguns", data=data, content_type="multipart/form-data")
        assert response.status_code == 200

        image_bytes = response.get_data()
        numpy_array = frombuffer(image_bytes, uint8)
        result_image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        cv2.imwrite(join(THIS_DIR, "output", "test_00.jpg"), result_image)

        # Test that having no file fails.
        response = client.post("/detect_handguns")
        assert response.status_code == 400
