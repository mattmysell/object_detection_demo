# Apply Machine Learning for Object Detection

For this project we have selected a publically available image dataset of handguns to create a handgun detection service. The goal is to detect handguns in a scene, specifically for security purposes, such as a handgun laying about or when a person is carrying or holding one.

The selection of a handgun dataset was for multiple reasons:

- It was publically available, thanks to the [University of Granada](https://sci2s.ugr.es/weapons-detection#Public%20datasets)
- It was already imported to RoboFlow, I beleive thanks for this goes to Alaa Sinjab, who has an excellent [tutorial](https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d) on how to apply this dataset. I will be noting the steps I take but for a more in depth tutorial you should read Alaa's
- And lastly, this is a large enough dataset with what appears to be mostly relevant/clean images that are appropriatly tagged so we are likely to get a good result without a huge amount of repetitive work

## Creating a Model

The model has already been generated and can be found at 'machine_learning/models/handguns/2', if you wish to generate yrouself it then the following outlines the steps taken, but for some more in depth guides on training object detection models I'd highly recommend these articles in addition to Alaa's above:

- [The practical guide for Object Detection with YOLOv5 algorithm](https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843) by Lihi Gur Arie, PhD
- [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results) by Glen Jocker

### Prepare the Dataset

After selecting the dataset, we are going to bring it into our own project on RoboFlow to make some further adjustments and prepare it for machine learning.

1. **Create a public/free account on [RoboFlow](https://roboflow.com/)**
2. **Fork the Pistols Dataset**
    - Once logged into RoboFlow fork the [Handguns Dataset](https://public.roboflow.com/object-detection/pistols) a button should appear in the top right allowing you to fork
3. **Generate the dataset**
    - Check the Train/Test split of images is close to 70% training, 20% validifying, and 10% testing. Adjust if necessary
    - Make sure to resize the images to a standard size, we will be using YoloV8 which is best with square images, and our images are mostly 500x360 so we'll pick 480x480 for the resize
    - We'll skip the augmentation for now as incorrectly augmenting the images can result in a worse outcome, let's just keep it simple
4. **Export the dataset**
    - Select the latest version of the dataset and being the export
    - Format will be YoloV8
    - Select "show download code"
    - Deselect "Also train a model for Label Assist with Roboflow Train." as this will use up your training credits
    - The Jupyter snippet will be copied over into training the model, so keep that ready

### Train the Model

For training Google Colab was used, to provide a powerful environment for free or a low cost. If you can I would recommend temporarily subscribing to Colab to gain access to the V1 and A1 servers and credits, it will significantly speed up training times and your sessions will be stored, although it is not necessary.

1. **Ensure you have an account for [Google Colab](https://colab.google/) and [Google Drive](https://www.google.com/drive/)**
2. **Open Colab and upload the notebook**
    - Once you are in Colab select File -> Upload Notebook, and select the [Training Notebook](handgun_detection_yolov8_training.ipynb)
3. **Follow the notebook steps**
    - Follow the steps in the Notebook, especially for loading in the Dataset as you will need the Jupyter snippet from [Part 4 of Preparing the Dataset](#prepare-the-dataset)
4. **Export the model**
    - After completing the steps in the Notebook on Google Colab we can download the openvino model
    - Create a new folder named '3' in 'models/handguns/'
    - Locate the openvino model files in your Google Drive, from training, and copy them into 'models/handguns/3'
    - Rename 'best.bin' and 'best.xml' to 'handguns.bin' and 'handguns.xml', leave metadata.yaml as is
    - Follow the instrustions for [Running Locally](../README.md#running-locally) to use your new model

### Enhance the Dataset

The following steps are optional in Roboflow for preparing the Dataset, but will get a better result.

1. **Clean the images in the dataset**
    - We are only interested in real images, so delete any images from the project that are drawings or cartoons. Photos of real handguns in pictures such as magazines are okay as that is still realistic enough for our objective
    - We don't want images that are far too close to the handguns, where most of the image is a portion of the handgun (less than 50% for example) as our aim is handguns in a scene not handguns in an abstract photograph, so delete these as well
    - A note about the dataset, is that it seems to have a lot more images of handguns on their own as opposed to being carried or held by a person, so our resulting object detection may perform better with handguns on their own
2. **Clean the tagging in the dataset**
    - For the best result if there are multiple handguns in an image all handguns you can make out should be tagged individually
    - Handguns should be consistently tagged, make sure you have a tight bounding box and if a handgun is partially visible try to bound the hidden parts as well
3. **Add null example images to the dataset**
    - Ideally we want to have around 10% of the images in the dataset having no handguns in them to aid the object detection in knowing what is not a handgun
    - The handguns dataset is comprised of images with handguns; on blank backgrounds, on tables, being held by people and a few other situations, therefore we would want null examples in similar sceneraios
        - We'll start with searching [Roboflow Universe](https://universe.roboflow.com/) for datasets containing either office stationary (for objects on blank backgrounds/tables that are not handguns), or poeple without handguns.
        - After finding some decent looking datasets we'll copy 150 images for each of the 2 types (office stationary, and people without handguns) as we have around 3000 images in our handgun dataset and want roughly 10% null examples
        - To copy the images, select the images you want to add and clone them into the handguns dataset, selecting "Import Raw Images" as we don't want the annotations
        - Followed by assigning all the new images to be annotated, then when annotating use the mark as null button to do them all at once, before adding them to the dataset.
