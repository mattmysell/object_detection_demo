#!/usr/bin/env python3
"""
Code for testing use of Openvino Runtime with Exported YoloV8 model.

This was run with the following python libraries installed:
    opencv-python==4.8.0
    ultralytics==8.0.196
    onnx==1.15.0 onnxruntime==1.16.2
    openvino==2023.2.0 openvino-dev==2023.2.0

test_yolov8.py must be run before this file, then run from this directory as:
    python test_yolov8.py
"""
# import numpy as np
from os import makedirs

# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from openvino.runtime import Core

ie = Core()

model = ie.read_model(model="best_openvino_model/best.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output()

image = cv2.imread("../test.jpg")
# N, C, H, W = input_layer_ir.shape
N, C, H, W = 1, 1, 480, 480
# resized_image = cv2.resize(image, (W, H))
# input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
input_image = cv2.dnn.blobFromImage(image, 1/255, (W, H), [0,0,0], 1, crop=False)
output = compiled_model([input_image])[output_layer_ir]
output = cv2.transpose(output[0])

boxes = []
scores = []
class_ids = []

for row in output:
    # Each row is [x, y, width, height, probability class 0, probability class 1, ...]
    classes_scores = row[4:]
    (_min_score, max_score, _min_class_loc, (_x, max_class_index)) = cv2.minMaxLoc(classes_scores)
    if max_score >= 0.25:
        box = [row[0] - (0.5 * row[2]), row[1] - (0.5 * row[3]), row[2], row[3]]
        boxes.append(box)
        scores.append(max_score)
        class_ids.append(max_class_index)

[height, width, _] = image.shape
length = max((height, width))
scale = length/480

# Apply NMS (Non-maximum suppression)
RESULT_BOXES = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.6, 0.5)
for index in RESULT_BOXES:
    box = boxes[index]
    x, y = round(box[0] * scale), round(box[1] * scale)
    x_plus_w, y_plus_h = round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
    image = cv2.rectangle(cv2.UMat(image), (x, y), (x_plus_w, y_plus_h), (0, 0, 255), 8)
    image = cv2.putText(cv2.UMat(image), f"handgun {round(scores[index], 2)}", (x - 10, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 8)

makedirs("runs/openvino", exist_ok=True)
cv2.imwrite("runs/openvino/test_openvino_runtime.jpg", image)
