#!/usr/bin/env python3
"""
Code for running a web ui for the image segmentation with streamlit.
"""
# Standard Libraries
from io import BytesIO

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
MAX_FILE_SIZE = 5*1024*1024  # 5MB

def image_to_bytes(image: Image) -> bytearray:
    """
    Convert an image to bytes.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    byte_im = buffer.getvalue()
    return byte_im

def call_detection(image: Image) -> Image:
    """
    Call the handguns detection endpoint, using image in IMAGES at index.
    """
    files = {"file": image_to_bytes(image)}
    response = post(ENDPOINT, files=files, timeout=5)
    buffer = BytesIO()
    for chunk in response.iter_content(1024):
        if chunk:
            buffer.write(chunk)
    buffer.seek(0)
    result_image = Image.open(buffer)
    return result_image

def detect(upload: str, col1, col2):
    """
    Remove the background from an image.
    """
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = call_detection(image)
    col2.write("Result Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download result image", image_to_bytes(fixed), "detection.png", "image/png")

def main():
    """
    Main function to run for streamlit web interface.
    """
    st.set_page_config(
        layout="wide",
        page_title="Object Detection Demo",
        # page_icon=Image.open(""),
        initial_sidebar_state="expanded"
    )
    session_id = get_script_run_ctx().session_id
    LOGR.set_context(session_id)
    LOGR.info("Started new session!")

    # Custom CSS to hide the deploy button
    st.markdown(
        r"""
        <style>
        .stAppDeployButton { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("## Detect Handguns in an Image")
    st.write(":dog: Try uploading an image you wish to detect handguns in. Result images can be "
            "downloaded from the sidebar.")
    st.sidebar.write("## Upload and download :gear:")

    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            detect(my_upload, col1, col2)
    else:
        detect("./test_00.jpg", col1, col2)

if __name__ == "__main__":
    main()
