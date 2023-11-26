#!/usr/bin/env python3
"""
Code for exporting YoloV8 model to Openvino and running inference.

This was run with the following python libraries installed:
    ultralytics==8.0.196
    onnx==1.15.0 onnxruntime==1.16.2
    openvino==2023.2.0 openvino-dev==2023.2.0

Run from this directory as:
    python test_yolov8.py
"""
from shutil import copyfile

from ultralytics import YOLO

model = YOLO("best.pt")
model.predict("../test.jpg", save=True)

copyfile("../test.jpg", "../test_openvino.jpg")
model.export(format="openvino", dynamic=True)

openvino_model = YOLO("best_openvino_model/")
openvino_model.predict("../test_openvino.jpg", save=True)
