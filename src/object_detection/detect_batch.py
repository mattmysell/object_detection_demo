#!/usr/bin/env python3
"""
Code for running object detection on an image.
"""

# Standard Libraries
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

# Create connection to the model server
CLIENT_ADDRESS = "localhost:9000"

def get_stub(using_tls: bool=False) -> PredictionServiceStub:
    """
    The PredictionServiceStub provides access to machine-learned models loaded by model_servers.
    """
    if using_tls:
        raise NotImplementedError("TLS has not been implemented, for more information see: " +\
            "https://github.com/openvinotoolkit/model_server/blob/main/demos/face_detection/python/face_detection.py")

    channel = insecure_channel(CLIENT_ADDRESS)
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)

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

def detect_batch(images: Union[str, List[NDArray]], model_name: str, model_classes: List[str], model_shape: List[int],
                 batch_size: int=1) -> NDArray:
    """
    Detect the objects in an image and return an image with the results.
    """
    if not is_type_list(model_shape, int, (2, 3)):
        raise ValueError("Invalid model_shape provided, expected [int, int]")

    if len(images) == 0:
        # No images were provided, so return.
        return

    stub = get_stub()
    blobs, images = load_blobs(images, model_shape)

    for x in range(0, blobs.shape[0] - batch_size + 1, batch_size):
        batch_blobs = blobs[x:(x + batch_size)]
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.inputs["images"].CopyFrom(make_tensor_proto(batch_blobs, shape=batch_blobs.shape))

        output = stub.Predict(request, 10.0)
        output = next(iter(output.outputs.values()))
        output = cv2.transpose(make_ndarray(output)[0])

        #TODO: Understand how the output varies and should be processed for batches.
        for y in range(0, batch_blobs.shape[0]):  # Iterate over all responses from images in the batch.
            #TODO: work out how to load the right image when running multiple batches of multiple images.
            output_image = images[x + y]
            detections = Detections(model_classes, output, model_shape=model_shape)
            detections.apply_non_max_suppression()
            output_image = detections.draw(output_image)

            cv2.imwrite(f"./output/test_batch_{x}_{y}.jpg", output_image)

if __name__ == "__main__":
    detect_batch(["./images/test_00.jpg"], "handguns", ["handgun"], (480, 480), 1)

    # cv2.imwrite("./output/test_00_detect.jpg",
    #             detect("./images/test_00.jpg", "handguns", ["handgun"], (480, 480)))
