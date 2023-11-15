#!/usr/bin/env python3
"""
Code for the endpoints in this application.
"""
# Standard Libraries
from base64 import b64encode
from json import dumps
from PIL import Image

# Installed Libraries
from flask import Flask, request, jsonify

# Local Files

app = Flask("detect_pistols_demo")

@app.route("/detect_pistol", methods=["POST"])
def endpoint_detect_pistol():
    """
    endpoint_detect_pistol() --> Response
    """
    image_file = request.files["file"]
    img = Image.open(image_file.stream)

    data = image_file.stream.read()
    data = b64encode(data).decode()

    output_dict = {
        "msg": "success",
        "size": [img.width, img.height],
        "format": img.format,
        "img": data
    }

    return jsonify(output_dict)