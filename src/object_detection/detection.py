#
# Copyright (c) 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
sys.path.append("../../common/python")

import argparse
import cv2
import datetime
import grpc
import numpy as np
import os
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import print_statistics, prepare_certs


def load_image(file_path):
    img = cv2.imread(file_path)  # BGR color format, shape HWC
    # img = cv2.resize(img, (args['width'], args['height']))
    # img = img.transpose(2,0,1).reshape(1,3,args['height'],args['width'])
    img = cv2.dnn.blobFromImage(img, 1/255, (480, 480), [0,0,0], 1, crop=False)
    # change shape to NCHW
    return img

parser = argparse.ArgumentParser(description='Demo for face detection requests via TFS gRPC API analyses input images '
                                             'and saves images with bounding boxes drawn around detected faces. '
                                             'It relies on face_detection model...')

parser.add_argument('--input_images_dir', required=False, help='Directory with input images', default="./images")
parser.add_argument('--output_dir', required=False, help='Directory for storing images with detection results', default="./output")
parser.add_argument('--batch_size', required=False, help='How many images should be grouped in one batch', default=1, type=int)
parser.add_argument('--width', required=False, help='How the input image width should be resized in pixels', default=480, type=int)
parser.add_argument('--height', required=False, help='How the input image width should be resized in pixels', default=480, type=int)
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name',required=False, default='handguns', help='Specify the model name')
parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
parser.add_argument('--server_cert', required=False, help='Path to server certificate')
parser.add_argument('--client_cert', required=False, help='Path to client certificate')
parser.add_argument('--client_key', required=False, help='Path to client key')
args = vars(parser.parse_args())

address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

channel = None
if args.get('tls'):
    server_ca_cert, client_key, client_cert = prepare_certs(server_cert=args['server_cert'],
                                                            client_key=args['client_key'],
                                                            client_ca=args['client_cert'])
    creds = grpc.ssl_channel_credentials(root_certificates=server_ca_cert,
                                         private_key=client_key, certificate_chain=client_cert)
    channel = grpc.secure_channel(address, creds)
else:
    channel = grpc.insecure_channel(address)

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

files = os.listdir(args['input_images_dir'])
batch_size = args['batch_size']
print(files)


imgs = np.zeros((0,3,args['height'],args['width']), np.dtype('<f'))
for i in files:
    img = load_image(os.path.join(args['input_images_dir'], i))
    imgs = np.append(imgs, img, axis=0)  # contains all imported images

print('Start processing {} iterations with batch size {}'.format(len(files)//batch_size , batch_size))

iteration = 0
processing_times = np.zeros((0),int)

CLASSES = ["handgun"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bounding_box(draw_image, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        draw_image (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    draw_image = cv2.rectangle(cv2.UMat(draw_image), (x, y), (x_plus_w, y_plus_h), color, 2)
    draw_image = cv2.putText(cv2.UMat(draw_image), label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return draw_image

for x in range(0, imgs.shape[0] - batch_size + 1, batch_size):
    iteration += 1

    img = imgs[x:(x + batch_size)]
    print("\nRequest shape", img.shape)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = args['model_name']
    request.inputs["images"].CopyFrom(make_tensor_proto(img, shape=(img.shape)))

    start_time = datetime.datetime.now()
    result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
    end_time = datetime.datetime.now()

    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    output = next(iter(result.outputs.values()))

    output = make_ndarray(output)[0]
    output = cv2.transpose(output)
    print(type(output))
    print("Response shape", output.shape)
    for y in range(0, img.shape[0]):  # iterate over responses from all images in the batch
        img_out = img[y,:,:,:]

        print("image in batch item",y, ", output shape",img_out.shape)
        img_out = img_out.transpose(1,2,0)

        boxes = []
        scores = []
        class_ids = []

        for row in output:
            classes_scores = row[4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                # print(f"Score: {round(maxScore, 4)}")
                box = [row[0] - (0.5 * row[2]), row[1] - (0.5 * row[3]), row[2], row[3]]
                # box coords = x, y, width, height
                print(f"    Row: {[round(value, 2) for value in row]}")
                # print(f"    Box: {[round(value) for value in box]}")
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

                # img_out = cv2.rectangle(cv2.UMat(img_out),(x_min, y_min),(x_max, y_max),(0,0,255),1)
                # img_out = cv2.rectangle(cv2.UMat(img_out),(box[0], box[1]),(box[2], box[3]),(0,0,255),1)

        [height, width, _] = img_out.shape
        length = max((height, width))
        scale = length/480

         # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        print(f"result_boxes: {type(result_boxes)}")
        for index in result_boxes:
            box = boxes[index]
            img_out = draw_bounding_box(img_out, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

        # for i in range(0, 200*batch_size-1):  # there is returned 200 detections for each image in the batch
        #     detection = output[y, i]
        #     print(type(detection))
        #     print(detection[:10])
        #     print(len(detection))
        #     # each detection has shape 1,1,7 where last dimension represent:
        #     # image_id - ID of the image in the batch
        #     # label - predicted class ID
        #     # conf - confidence for the predicted class
        #     # (x_min, y_min) - coordinates of the top left bounding box corner
        #     #(x_max, y_max) - coordinates of the bottom right bounding box corner.
        #     if detection[0,0,2] > 0.5 and int(detection[0,0,0]) == y:  # ignore detections for image_id != y and confidence <0.5
        #         print("detection", i , detection)
        #         x_min = int(detection[0,0,3] * args['width'])
        #         y_min = int(detection[0,0,4] * args['height'])
        #         x_max = int(detection[0,0,5] * args['width'])
        #         y_max = int(detection[0,0,6] * args['height'])
        #         # box coordinates are proportional to the image size
        #         print("x_min", x_min)
        #         print("y_min", y_min)
        #         print("x_max", x_max)
        #         print("y_max", y_max)

        #         img_out = cv2.rectangle(cv2.UMat(img_out),(x_min,y_min),(x_max,y_max),(0,0,255),1)
                # draw each detected box on the input image
        print("saving result to",os.path.join(args['output_dir'],str(iteration)+"_"+str(y)+'.jpg'))
        cv2.imwrite(os.path.join(args['output_dir'],str(iteration)+"_"+str(y)+'.jpg'),img_out)

    print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'
          .format(iteration, round(np.average(duration), 2), round(1000 * batch_size / np.average(duration), 2)
                                                                                  ))

print_statistics(processing_times, batch_size)