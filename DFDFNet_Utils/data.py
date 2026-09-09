"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/data.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Ninapro dataset loading, preprocessing and splitting.
----------------------------------------------------------------------------------------------------
    Loads Ninapro sEMG data with one-hot encoding and MVC normalization,
    and splits the dataset by repetition or random split.

"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

__all__ = ["load_nina_pro", "split_dataset", "resolve_data_path"]


def resolve_data_path(source: str, subject: int, window_length: int) -> str:
    # Resolve the Ninapro data file path
    return os.path.join(
        source,
        f"Results_{window_length}ms",
        f"Ninapro_Data_S{subject}_A1_E123",
        "data_npy.npy",
    )


def _apply_process_method(x: np.ndarray, method: str) -> np.ndarray:
    # Apply the requested data preprocessing method
    if method in (None, "None", "none", ""):
        return x
    if method == "Reshape":
        return x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    if method in ("PSR", "GAF", "MTF", "RP"):
        raise NotImplementedError(
            f"Preprocessing method {method!r} depends on phase-space/image encoding modules, not available in this open-source version."
        )
    raise ValueError(f"Unknown preprocessing method: {method!r}")


def load_nina_pro(
    source: str,
    label_list: Optional[Iterable[int]] = None,
    subject_list: Optional[Iterable[int]] = None,
    channel_list: Optional[Iterable[int]] = None,
    window_length: int = 200,
    process_method: str = "None",
    is_mvc: bool = True,
    return_mvc: bool = False,
    mvc_max: Optional[np.ndarray] = None,
    mvc_min: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    # Load Ninapro sEMG data with one-hot encoding and MVC normalization
    subjects = list(subject_list) if subject_list else [1]
    xs, ys = [], []
    for subject in subjects:
        path = resolve_data_path(source, subject, window_length)
        with open(path, "rb") as f:
            x, y = np.load(f), np.load(f)
        if label_list:
            keep = np.isin(y[:, 0], list(label_list))
        else:
            keep = y[:, 0] != 0
        xs.append(x[keep])
        ys.append(y[keep])

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    y_onehot = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

    if channel_list:
        x = x[:, :, list(channel_list)]

    x = _apply_process_method(x, process_method)

    if is_mvc and process_method in (None, "None", "none", ""):
        x_max = mvc_max if mvc_max is not None else np.max(x, axis=0)
        x_min = mvc_min if mvc_min is not None else np.min(x, axis=0)
        x = (x - x_min) / (x_max - x_min)
        x = np.nan_to_num(x, nan=0.5)
        if return_mvc:
            print(f"➣ MVC global normalization info extracted, subjects: {subjects}")
            return {"max": x_max, "min": x_min}

    print(f"➣ Training data loaded ({source})")
    print(f"   X shape: {x.shape}   Y shape: {y_onehot.shape}")
    return x, y_onehot


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    val_index: Optional[list[int]] = None,
    train_index: Optional[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Split the dataset by action repetition index
    train_index = train_index if train_index is not None else [1, 3, 4, 6]
    val_index = val_index if val_index is not None else [2, 5]
    labels = np.argmax(y, axis=1)

    val_flags, train_flags = [], []
    for cls in np.unique(labels):
        indices = np.where(labels == cls)[0]
        span = len(indices) // (len(val_index) + len(train_index))
        val_span = [i for r in val_index for i in range((r - 1) * span, r * span)]
        train_span = [i for r in train_index for i in range((r - 1) * span, r * span)]
        val_flags.append(indices[val_span])
        train_flags.append(indices[train_span])

    val_flags = np.concatenate(val_flags)
    train_flags = np.concatenate(train_flags)
    return x[train_flags], x[val_flags], y[train_flags], y[val_flags]


def random_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Split the dataset randomly into train / validation sets
    return train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)
