#!/usr/bin/env python3
"""
Code for running object detection on batches of images, more suitable for GPUs.

As we are running inside a docker container there are extra steps to get the most out of tensorflow, specifially using
Cuda and running on an NVIDIA GPU. For now we won't go through all those steps as we do not have a clear production
environment in mind, so will stick to just showing how tensorflow can be used without optimization.
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
from object_detection.detections import is_type_list, Detections
from object_detection.metadata import ModelMetadata

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

def batch_inference(blobs: NDArray, model_meta: ModelMetadata, batch_size: int) -> List[Detections]:
    """
    Perfrom the inference on the blobs in batches.
    """
    detections = []
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_meta.name
    processed = 0
    while processed < blobs.shape[0]:
        batch_blobs = blobs[processed:(processed + batch_size)]
        request.inputs["images"].CopyFrom(make_tensor_proto(batch_blobs, shape=batch_blobs.shape))
        batch_output = STUB.Predict(request, 10.0)
        batch_output = make_ndarray(next(iter(batch_output.outputs.values())))

        for y in range(batch_output.shape[0]):  # Iterate over all responses from images in the batch.
            output = cv2.transpose(batch_output[y])
            detections.append(Detections(model_meta.classes, output, model_shape=model_meta.input_shape))
            detections[-1].apply_non_max_suppression()
        processed += batch_size
    return detections

def detect_batch(images: Union[str, List[NDArray]], model_meta: ModelMetadata, batch_size: int=1) -> Union[float, None]:
    """
    Detect the objects in an image and return an image with the results.
    """
    if not is_type_list(model_meta.input_shape, int, exact_type=True, list_length=(2, 3)):
        raise ValueError("Invalid model_shape provided, expected [int, int]")

    if len(images) == 0:
        # No images were provided, so return.
        return None

    blobs, images = load_blobs(images, model_meta.input_shape)
    detections = []

    inference_start = perf_counter()
    detections = batch_inference(blobs, model_meta, batch_size)
    inference_end = perf_counter()

    output_images = []
    for i, output_image in enumerate(images):
        output_image = detections[i].draw(output_image)
        output_images.append(output_image)

    return output_images, inference_end - inference_start
