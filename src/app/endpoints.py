#!/usr/bin/env python3
"""
Code for the endpoints in this application.
"""
# Standard Libraries
# from base64 import b64encode
# from PIL import Image

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from flask import Flask, request, Response
from numpy import frombuffer, uint8

# Local Files
from object_detection.detect import detect

app = Flask("object_detection_demo")

@app.route("/detect_handguns", methods=["POST"])
def endpoint_detect_handguns() -> Response:
    """
    Endpoint for detecting handguns in an image and returning an image with the identified objects.
    """
    image_file = request.files["file"]

    numpy_array = frombuffer(image_file.stream.read(), uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

    result_image, _time = detect(image, "handguns")

    _retval, buffer = cv2.imencode('.jpg', result_image)

    return Response(buffer.tobytes(), status=200, mimetype="image/jpeg")
