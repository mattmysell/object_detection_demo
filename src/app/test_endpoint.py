#!/usr/bin/env python3
"""
Code for unit testing.
"""
# Standard Libraries
from io import BytesIO
from os.path import dirname, join, realpath

# Installed Libraries

# Local Files
from endpoints import app

THIS_DIR = dirname(realpath(__file__))

def test_endpoint_detect_pistol():
    """
    test_endpoint_detect_pistol() --> null
    """
    with open(join(THIS_DIR, "test_image_00.jpg"), "rb") as image_file:
        test_image = image_file.read()

    with app.test_client() as client:
        data = {}
        data["file"] = (BytesIO(test_image), "image.jpg")
        response = client.post("/detect_pistol", data=data, content_type="multipart/form-data")
        assert response.status_code == 200

        # Test that having no file fails.
        response = client.post("/detect_pistol")
        assert response.status_code == 400
