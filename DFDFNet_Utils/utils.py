"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/utils.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    General utility functions.
----------------------------------------------------------------------------------------------------
    Provides utilities for random seeding, log-level configuration,
    environment reporting, directory management, and duration formatting.

"""

from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np
import tensorflow as tf

__all__ = [
    "set_seed",
    "set_tf_log_level",
    "ensure_dir",
    "print_environment",
    "configure_console_encoding",
    "print_device_summary",
    "format_duration",
]


def configure_console_encoding() -> None:
    # Reconfigure stdout/stderr to UTF-8 to avoid GBK encoding errors on Windows
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None) or ""
        if encoding.lower() not in ("utf-8", "utf8") and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def set_seed(seed: int = 1) -> None:
    # Set the global random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_tf_log_level(mode: int = 2) -> None:
    # Set the TensorFlow logging verbosity level (0 all, 1 no info, 2 no warning, 3 no error)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(mode)


def ensure_dir(path: str) -> str:
    # Recursively create the directory if it does not exist
    os.makedirs(path, exist_ok=True)
    return path


def _print_section(title: str) -> None:
    # Print a section separator title
    print("\n" + "=" * 24 + f" {title} " + "=" * 24)


def print_environment() -> None:
    # Print OS, Python and training resource information
    _print_section("OS Info")
    print(f"Computer name: {platform.node()}")
    print(f"System name: {platform.system()}")
    print(f"System release: {platform.release()}")
    print(f"CPU arch: {platform.machine()}")

    _print_section("Python Environment Info")
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python implementation: {platform.python_implementation()}")

    _print_section("Training Resources")
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    if gpus:
        for index, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            print(f"({index}) GPU: {details.get('device_name', 'Unknown GPU')}")
            print(f"    GPU compute capability: {details.get('compute_capability', 'Unknown')}")
    elif cpus:
        print(f"CPU model: {platform.processor()}")
        print(f"CPU count: {len(cpus)}")
    else:
        print("No available GPU or CPU resources!")


def print_device_summary() -> None:
    # Print the GPU / CPU device configuration used for training
    _print_section("Training Device Config")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"Training device: GPU ({details.get('device_name', 'Unknown GPU')})")
            print(f"    GPU compute capability: {details.get('compute_capability', 'Unknown')}")
    else:
        cpus = tf.config.list_physical_devices("CPU")
        print(f"Training device: CPU ({platform.processor() or 'Unknown CPU'})")
        print(f"    CPU count: {len(cpus)}")


def format_duration(seconds: float) -> str:
    # Format seconds into a readable ``Xh Ym Zs`` duration string
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
