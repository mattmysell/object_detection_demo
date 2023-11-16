# Apply Machine Learning for Object Detection

For this project we have selected a publically available image dataset of handguns to create a handgun detection service. The goal is to detect handguns in a scene, specifically for security purposes, so of a handgun laying about or more so when a person is carrying/holding one.

The selection of a handgun dataset was for multiple reasons:

- It was publically available, thanks to the [University of Granada](https://sci2s.ugr.es/weapons-detection#Public%20datasets)
- It was already imported to RoboFlow, I beleive thanks for this goes to Alaa Sinjab, who has an excellent [tutorial](https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d) on how to apply this dataset. I will be noting the steps I take but for a more in depth tutorial you should read Alaa's
- And lastly, this is a large enough dataset with what appears to be mostly relevant/clean images that are appropriatly tagged so we are likely to get a good result without a huge amount of repetitive work

## Prepare the Dataset

After selecting the dataset, we are going to bring it into our own project on RoboFlow to make some further adjustments and prepare it for machine learning.

I will be outlining the steps taken, but for some more in depth guides on training object detection models I'd highly recommend these articles in addition to Alaa's above:

- [The practical guide for Object Detection with YOLOv5 algorithm](https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843) by Lihi Gur Arie, PhD
- [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results) by Glen Jocker

### Steps

1. **Create a public/free account on [RoboFlow](https://roboflow.com/)**
2. **Fork the Pistols Dataset**
    - Once logged into RoboFlow fork the [Handguns Dataset](https://public.roboflow.com/object-detection/pistols) a button should appear in the top right allowing you to fork
3. **Generate the dataset**
    - Check the Train/Test split of images is close to 70% training, 20% validifying, and 10% testing. Adjust if necessary
    - Make sure to resize the images to a standard size, we will be using YoloV5 which is best with square images, and our images are mostly 500x360 so we'll pick 480x480 for the resize
    - We'll skip the augmentation for now as incorrectly augmenting the images can result in a worse outcome, let's just keep it simple
4. **Export the dataset**
    - Select the latest version of the dataset and being the export
    - Format will be YoloV5 PyTorch
    - Download zip to computer
    - Deselect "Also train a model for Label Assist with Roboflow Train." as this will use up your training credits
    - After the download you can move the zip file to this directory

## Train the Model



## Enhance the Model

3. **Clean the images in the dataset**, this is optional but will give you a better result
    - We are only interested in real images, so delete any images from the project that are drawings or cartoons. Photos of real handguns in pictures such as magazines are okay as that is still realistic enough for our objective
    - We don't want images that are far too close to the handguns, where most of the image is a portion of the handgun (less than 50% for example) as our aim is handguns in a scene not handguns in an abstract photograph, so delete these as well
    - A note about the dataset, is that it seems to have a lot more images of handguns on their own as opposed to being carried or held by a person, so our resulting object detection may perform better with handguns on their own
4. **Clean the tagging in the dataset**, this is optional but will give you a better result
    - For the best result if there are multiple handguns in an image all handguns you can make out should be tagged individually
    - Handguns should be consistently tagged, make sure you hve a tight bounding box and if a handgun is partially visible try to bound the hidden parts as well
    - This dataset is already tagged well, so we don't have to fix anything up
5. **Add null example images to the dataset**, this is optional but will give you a better result
    - Ideally we want to have around 10% of the images in the dataset having no handguns in them to aid the object detection in knowing what is not a handgun
    - The handguns dataset is comprised of images with handguns; on blank backgrounds, on tables, being held by people and a few other situations, therefore we would want null examples in similar sceneraios
        - We'll start with searching [Roboflow Universe](https://universe.roboflow.com/) for datasets containing either office stationary (for objects on blank backgrounds/tables that are not handguns), or poeple without handguns.
        - After finding some decent looking datasets we'll copy 150 images for each of the 2 types (office stationary, and people without handguns) as we have around 3000 images in our handgun dataset and want roughly 10% null examples
        - To copy the images, select the images you want to add and clone them into the handguns dataset, selecting "Import Raw Images" as we don't want the annotations
        - Followed by assigning all the new images to be annotated, then when annotating use the mark as null button to do them all at once, before adding them to the dataset