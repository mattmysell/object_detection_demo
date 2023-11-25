#!/usr/bin/env python3
"""
Code for running object detection on batches of images, more suitable for GPUs.
"""

# Standard Libraries
from os import environ
from time import perf_counter
from typing import List, Tuple, Union

# Installed Libraries
# pylint: disable=no-member
import cv2 # We have to disable no member as pylint is not aware of cv2s members.
from grpc import insecure_channel
import numpy as np
from numpy.typing import NDArray
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

# Local Files
from detections import is_type_list, Detections
from utils import print_statistics

# Create connection to the model server
HOST = environ.get("INFERENCE_HOST", "localhost")
PORT = environ.get("INFERENCE_PORTT", "9000")

def get_stub(using_tls: bool=False) -> PredictionServiceStub:
    """
    The PredictionServiceStub provides access to machine-learned models loaded by model_servers.
    """
    if using_tls:
        raise NotImplementedError("TLS has not been implemented, for more information see: " +\
            "https://github.com/openvinotoolkit/model_server/blob/main/demos/face_detection/python/face_detection.py")

    channel = insecure_channel(f"{HOST}:{PORT}")
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)

STUB = get_stub()

def load_blobs(images: Union[str, List[NDArray]], model_shape: List[int]) -> Tuple[NDArray, List[NDArray]]:
    """
    Load the images in, if required, and convert to blobs.
    """
    blobs = np.zeros((0, 3, model_shape[0] ,model_shape[1]), np.dtype('<f'))
    if len(images) == 0:
        return blobs, []

    for i, image in enumerate(images):
        if isinstance(image, str):
            images[i] = cv2.imread(image)
        image_blobs = cv2.dnn.blobFromImage(images[i], 1/255, model_shape, [0,0,0], 1, crop=False)
        blobs = np.append(blobs, image_blobs, axis=0)
    return blobs, images

def batch_inference(blobs: NDArray, model_name: str, model_classes: List[str], model_shape: List[int],
                     batch_size: int) -> List[Detections]:
    """
    Perfrom the inference on the blobs in batches.
    """
    detections = []
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    for x in range(0, blobs.shape[0] - batch_size + 1, batch_size):
        batch_blobs = blobs[x:(x + batch_size)]
        request.inputs["images"].CopyFrom(make_tensor_proto(batch_blobs, shape=batch_blobs.shape))
        batch_output = STUB.Predict(request, 10.0)
        batch_output = make_ndarray(next(iter(batch_output.outputs.values())))

        for y in range(batch_output.shape[0]):  # Iterate over all responses from images in the batch.
            output = cv2.transpose(batch_output[y])
            detections.append(Detections(model_classes, output, model_shape=model_shape))
            detections[-1].apply_non_max_suppression()
    return detections

def detect_batch(images: Union[str, List[NDArray]], model_name: str, model_classes: List[str], model_shape: List[int],
                 batch_size: int=1) -> Union[float, None]:
    """
    Detect the objects in an image and return an image with the results.
    """
    if not is_type_list(model_shape, int, (2, 3)):
        raise ValueError("Invalid model_shape provided, expected [int, int]")

    if len(images) == 0:
        # No images were provided, so return.
        return None

    blobs, images = load_blobs(images, model_shape)
    detections = []

    inference_start = perf_counter()
    detections = batch_inference(blobs, model_name, model_classes, model_shape, batch_size)
    inference_end = perf_counter()

    for i, output_image in enumerate(images):
        output_image = detections[i].draw(output_image)
        cv2.imwrite(f"./output/test_{str(i).zfill(2)}_detect_batch.jpg", output_image)

    return inference_end - inference_start

if __name__ == "__main__":
    input_images = [f"./images/test_{str(i).zfill(2)}.jpg" for i in range(6)]
    inference_seconds = detect_batch(input_images, "handguns", ["handgun"], (480, 480), 3)
    print_statistics([inference_seconds*1000], 1)
