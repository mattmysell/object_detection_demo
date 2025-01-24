#!/usr/bin/env python3
"""
Code for running a web ui for the image segmentation with streamlit.
"""
# Standard Libraries
from functools import partial
from io import BytesIO
from time import time
from typing import Tuple

# Installed Libraries
from PIL import Image
from requests import post
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Local Files
from utils import ContextLogger

LOGR = ContextLogger("demo_ui")

# Port should be the internal port used in the app, see src/main.py
ENDPOINT = "http://app:5000/detect_handguns"
MAX_FILE_SIZE = 10*1024*1024  # 5MB

def image_to_bytes(image: Image) -> bytearray:
    """
    Convert an image to bytes.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    byte_im = buffer.getvalue()
    return byte_im

def call_detection(image: Image) -> Tuple[Image.Image, float]:
    """
    Call the handguns detection endpoint, using image in IMAGES at index.
    """
    start_time = time()
    files = {"file": image_to_bytes(image)}
    response = post(ENDPOINT, files=files, timeout=5)
    buffer = BytesIO()
    for chunk in response.iter_content(1024):
        if chunk:
            buffer.write(chunk)
    buffer.seek(0)
    result_image = Image.open(buffer)
    processing_time = round(time() - start_time, 4)
    LOGR.info("Processed image in %s seconds", processing_time)
    return result_image, processing_time

def detect(upload: str, empty_1: st.container, empty_2: st.container):
    """
    Remove the background from an image.
    """
    LOGR.info("Running detection on: %s", upload)
    st.session_state["original"] = Image.open(upload)
    st.session_state["result"], st.session_state["processing_time"] = call_detection(st.session_state["original"])
    display(empty_1, empty_2)

def display(empty_1: st.container, empty_2: st.container):
    """
    Display images and download button.
    """
    container_1, container_2 = empty_1.container(), empty_2.container()
    width, height = st.session_state["original"].size
    container_1.write(f"**Original {width}x{height} Image**")
    container_1.image(st.session_state["original"])
    container_2.write(f"**Result in {st.session_state['processing_time']} seconds**")
    container_2.image(st.session_state["result"])
    LOGR.info("Updated display images")

def uploader_change():
    """
    When the uploader changes set use upload to True, even if the upload is removed.
    """
    st.session_state["run_detection_on_upload"] = True
    LOGR.info("Uploade change has occured")

def main():
    """
    Main function to run for streamlit web interface.
    """
    st.set_page_config(
        layout="wide",
        page_title="Object Detection Demo",
        page_icon=Image.open("./icon.png"),
        initial_sidebar_state="expanded"
    )
    session_id = get_script_run_ctx().session_id
    LOGR.set_context(session_id)
    LOGR.info("In session: %s", session_id)

    # Custom CSS to hide the deploy button
    st.markdown(
        r"""
        <style>
        .stAppDeployButton { visibility: hidden; }
        section[data-testid="stSidebar"] div.stButton button {
            color: GhostWhite;
            background-color: Olive;
            width: 200px;
        }
        section[data-testid="stSidebar"] div.stDownloadButton button {
            color: GhostWhite;
            background-color: #808080;
            width: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set session variables
    if "original" not in st.session_state:
        st.session_state["original"] = Image.open("samples/test_04.jpg")
    if "result" not in st.session_state:
        result, processing_time = call_detection(st.session_state["original"])
        st.session_state["result"] = result
        st.session_state["processing_time"] = processing_time
    if "run_detection_on_upload" not in st.session_state:
        st.session_state["run_detection_on_upload"] = True

    st.write("## Detect Handguns in an Image")
    st.write("Try uploading an image you wish to detect handguns in. Result images can be "
            "downloaded from the sidebar.")

    column_1, column_2 = st.columns(2)
    # Placeholders to call when we want to refresh the containers
    empty_1, empty_2 = column_1.empty(), column_2.empty()

    st.sidebar.write("### Upload an Image")
    my_upload = st.sidebar.file_uploader("upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
                                         on_change=uploader_change)

    samples = {
        "samples/test_06.jpg": "Over the Shoulder",
        "samples/test_04.jpg": "Handguns on Table",
        "samples/test_01.jpg": "Home Intruder",
        "samples/test_00.jpg": "Single Handgun",
        "samples/test_09.jpg": "Dual Wielding",
    }
    st.sidebar.markdown("\n\n### Additional Samples")
    sample_button_clicked = []
    for file_path, description in samples.items():
        sample_button_clicked.append(
            st.sidebar.button(description.ljust(20), on_click=partial(detect, file_path, empty_1, empty_2)))

    if (my_upload is not None) and st.session_state["run_detection_on_upload"]:
        # Do not run detection on upload if the session reruns.
        st.session_state["run_detection_on_upload"] = False
        if my_upload.size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload an image smaller than 10MB.")
        else:
            detect(my_upload, empty_1, empty_2)

    display(empty_1, empty_2)
    st.sidebar.markdown("\n\n### Download")
    st.sidebar.download_button("Download Result Image",
                               image_to_bytes(st.session_state["result"]), "detection.png", "image/png")

if __name__ == "__main__":
    main()
