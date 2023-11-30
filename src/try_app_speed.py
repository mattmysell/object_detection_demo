#!/usr/bin/env python3
"""
Code for testing speed of the app when running object detection many images.

To be run from the your PC (not in the docker containers of this project).

Note that varying the number of threads for the speed test will alter the total fps, try altering the number of threads
in ThreadPool on line 81 to see what works best for you.
"""
# Standard Libraries
from multiprocessing.pool import ThreadPool
from os import makedirs
from os.path import dirname, join, pardir, realpath
from random import shuffle
from time import time
from typing import List

# Installed Libraries
from requests import get

# Local Files
from object_detection.utils import print_statistics

ENDPOINT = "http://localhost:8080/detect_handguns"

THIS_DIR = dirname(realpath(__file__))
TEST_FILES_DIR = join(THIS_DIR, pardir, "test_files")
INPUT_DIR = join(TEST_FILES_DIR, "input")
OUTPUT_DIR = join(TEST_FILES_DIR, "output", "try_app_speed")
makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_COUNT = 10
IMAGES = []
for i in range(IMAGE_COUNT):
    with open(join(INPUT_DIR, f"test_{str(i).zfill(2)}.jpg"), "rb") as image_file:
        IMAGES.append(image_file.read())

def call_detection(inputs: List[any], write_image: bool=False):
    """
    Call the handguns detection endpoint, using image in IMAGES at index.
    """
    [test_index, image_index, results_store, milliseconds_store] = inputs

    start_time = time()
    files = {"file": IMAGES[image_index]}
    response = get(ENDPOINT, files=files, timeout=5)
    request_duration = time() - start_time
    milliseconds_store[test_index] = request_duration*1000

    if response.status_code == 200:
        if write_image:
            with open(join(OUTPUT_DIR, f"test_{str(i).zfill(2)}.jpg"), "wb") as result_image:
                for chunk in response.iter_content(1024):
                    result_image.write(chunk)
        results_store[test_index] = True
    else:
        results_store[test_index] = False

if __name__ == "__main__":
    # Check that the inference is working.
    results = {}
    for i in range(0, IMAGE_COUNT):
        call_detection([i, i, results, {}], True)
    successes = sum(results.values())

    if successes != IMAGE_COUNT:
        print(f"\n{successes} of {IMAGE_COUNT} results ran successfully, check to see issues before proceeding.")
        print("    See resulting output images at: test_files/output/try_app_speed/")
    else:
        print(f"\nAll {IMAGE_COUNT} results ran successfully, continuing to check speed...")
        print("    See resulting output images at: test_files/output/try_app_speed/")

        results, milliseconds = {}, {}
        test_indices = list(range(IMAGE_COUNT))*10
        shuffle(test_indices)
        test_indices = [
            [test_index, image_index, results, milliseconds] for test_index, image_index in enumerate(test_indices)]

        start = time()
        # Varying the number of threads will affect the overall efficiency, for my setup 3 threads is optimal.
        with ThreadPool(3) as thread_pool:
            thread_pool.map(call_detection, test_indices)
        total_duration = round(time() - start, 2)
        average_fps = round(len(test_indices)/total_duration, 2)

        print_statistics("Calling App", list(milliseconds.values()), len(test_indices))

        successes = sum(results.values())
        if successes != len(test_indices):
            print(f"\n {successes} of {len(test_indices)} speed test "
                  f"results ran successfully in {total_duration} seconds, {average_fps} fps")
        else:
            print(f"\n All {len(test_indices)} speed test "
                  f"results ran successfully in {total_duration} seconds, {average_fps} fps")
