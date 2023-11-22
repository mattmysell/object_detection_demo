import cv2
from ovmsclient import make_grpc_client
from time import time

# Create connection to the model server
client = make_grpc_client("localhost:9000")

# # Get model metadata to learn about model inputs
# model_metadata = client.get_model_metadata(model_name="handguns", model_version=1, timeout=30.0)

# # If model has only one input, get its name
# input_name = next(iter(model_metadata["inputs"]))

CLASSES = ["handgun"]
colors = [(0, 0, 255)] # np.random.uniform(0, 255, size=(len(CLASSES), 3))

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
    label = f'{CLASSES[class_id]} {confidence:.2f}'
    color = colors[class_id]
    draw_image = cv2.rectangle(cv2.UMat(draw_image), (x, y), (x_plus_w, y_plus_h), color, 8)
    draw_image = cv2.putText(cv2.UMat(draw_image), label, (x - 10, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)
    return draw_image

# Read the image file
original = cv2.imread("./images/test_image_00.jpg")
# Resize
# formatted = cv2.resize(original, (480, 480))
# Change shape to NCHW
# formatted = formatted.transpose(2,0,1).reshape(1,3,480,480)
# formatted = formatted.astype(np.float32)
# formatted /= 255.

# formatted = np.expand_dims(formatted, 0)  # Add batch dimension.
formatted = cv2.dnn.blobFromImage(original, 1/255, (480, 480), [0,0,0], 1, crop=False)
print(formatted.shape)
# with open("./images/test_image_00.jpg", "rb") as input_image:
#     original = input_image.read()

# Place the data in a dict, along with model input name
inputs = {"images": formatted}

# Run prediction and wait for the result
start = time()
results = client.predict(inputs=inputs, model_name="handguns")
end = time()
output = results[0]
output = cv2.transpose(output)
print(type(output))
print("Response shape", output.shape)
# original = img[y,:,:,:]

# print("image in batch item",y, ", output shape",original.shape)
# original = original.transpose(1,2,0)

boxes = []
scores = []
class_ids = []

for row in output:
    classes_scores = row[4:]
    (_minScore, maxScore, _minClassLoc, (_x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
    if maxScore >= 0.25:
        # print(f"Score: {round(maxScore, 4)}")
        box = [row[0] - (0.5 * row[2]), row[1] - (0.5 * row[3]), row[2], row[3]]
        # box coords = x, y, width, height
        print(f"    Row: {[round(value, 2) for value in row]}")
        # print(f"    Box: {[round(value) for value in box]}")
        boxes.append(box)
        scores.append(maxScore)
        class_ids.append(maxClassIndex)

        # original = cv2.rectangle(cv2.UMat(original),(x_min, y_min),(x_max, y_max),(0,0,255),1)
        # original = cv2.rectangle(cv2.UMat(original),(box[0], box[1]),(box[2], box[3]),(0,0,255),1)

[height, width, _] = original.shape
length = max((height, width))
scale = length/480

# Apply NMS (Non-maximum suppression)
RESULT_BOXES = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

print(f"RESULT_BOXES: {type(RESULT_BOXES)}")
for index in RESULT_BOXES:
    box = boxes[index]
    original = draw_bounding_box(
        original, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
        round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

print("saving result to","./images/test_image_00_detect.jpg")
cv2.imwrite("./images/test_image_00_detect.jpg", original)

print(f"Inference time: {round(1000*(end-start))} ms")