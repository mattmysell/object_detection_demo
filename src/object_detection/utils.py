#!/usr/bin/env python3
"""
Utility code for other files to use within this folder.
"""
# Standard Libraries
from typing import List

# Installed Libraries
import numpy as np

# Local Files

def print_statistics(inference_milliseconds: List[float], batch_size: int):
    """
    Print out the statistics from a process that has been run.
    """
    print("\nProcessing time for all iterations")
    print(f"    Average time: {np.average(inference_milliseconds):.2f} ms; "
          f"Average speed: {1000*batch_size/np.average(inference_milliseconds):.2f} fps")
    print(f"    Median time: {np.median(inference_milliseconds):.2f} ms; "
          f"Median speed: {1000*batch_size/np.median(inference_milliseconds):.2f} fps")
    print(f"    Max time: {np.max(inference_milliseconds):.2f} ms; "
          f"Min speed: {1000*batch_size/np.max(inference_milliseconds):.2f} fps")
    print(f"    Min time: {np.min(inference_milliseconds):.2f} ms; "
          f"Max speed: {1000*batch_size/np.min(inference_milliseconds):.2f} fps")
    print(f"    Time percentile 90: {np.percentile(inference_milliseconds, 90):.2f} ms; "
          f"Speed percentile 90: {1000*batch_size/np.percentile(inference_milliseconds, 90):.2f} fps")
    print(f"    Time percentile 50: {np.percentile(inference_milliseconds, 50):.2f} ms; "
          f"Speed percentile 50: {1000*batch_size/np.percentile(inference_milliseconds, 50):.2f} fps")
    print(f"    Time standard deviation: {np.std(inference_milliseconds):.2f}")
    print(f"    Time variance: {np.var(inference_milliseconds):.2f}")
