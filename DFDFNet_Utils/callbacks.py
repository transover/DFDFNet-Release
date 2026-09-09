"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/callbacks.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Training callbacks and checkpoint resume.
----------------------------------------------------------------------------------------------------
    Builds training callbacks (checkpoint, reduce-LR, early stopping)
    and handles the checkpoint resumption logic.

"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional

from tensorflow import keras

from .config import HyperParameters
from .utils import ensure_dir

__all__ = ["build_callbacks", "prepare_checkpoint", "find_latest_checkpoint"]


def checkpoint_path(model_name: str, output_dir: str = "models") -> str:
    # Build the model weights checkpoint path template
    return os.path.join(output_dir, model_name, "checkpoints", "{epoch:02d}-{val_loss:.2f}.weights.h5")


def build_callbacks(hp: HyperParameters, model_name: str, output_dir: str = "models") -> list:
    # Build the list of training callbacks (checkpoint, reduce LR, early stopping)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path(model_name, output_dir),
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        mode="auto",
        save_freq="epoch",
        verbose=1,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=hp.lr_factor,
        patience=hp.lr_patience,
        min_lr=hp.lr_min,
        verbose=1,
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=hp.patience,
        restore_best_weights=True,
        verbose=1,
    )
    return [checkpoint, reduce_lr, early_stop]


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[Optional[str], Optional[int], Optional[float]]:
    # Find the latest model weights record in the directory
    if not os.path.isdir(checkpoint_dir):
        return None, None, None

    files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".weights.h5"))
    if not files:
        return None, None, None

    latest = files[-1]
    match = re.search(r"^(\d+)-(\d+\.\d+)\.weights\.h5$", latest)
    path = os.path.join(checkpoint_dir, latest)
    if match:
        return path, int(match.group(1)), float(match.group(2))
    return path, None, None


def prepare_checkpoint(
    model: keras.Model,
    reload_checkpoint: bool,
    model_name: str,
    output_dir: str = "models",
) -> int:
    # Handle checkpoint resume logic and return the initial epoch
    ckpt_dir = os.path.dirname(checkpoint_path(model_name, output_dir))
    initial_epoch = 0

    if reload_checkpoint:
        latest_path, latest_epoch, latest_val_loss = find_latest_checkpoint(ckpt_dir)
        if latest_path and os.path.exists(latest_path):
            model.load_weights(latest_path)
            initial_epoch = latest_epoch or 0
            print(
                f"\n[Resume OK] Checkpoint loaded, Epoch:{latest_epoch} Val_loss:{latest_val_loss} [{latest_path}]\n"
            )
        else:
            print("\n[Resume Failed] No trained model found under checkpoints directory\n")
    else:
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        ensure_dir(ckpt_dir)
        print(f"\n[Resume Disabled] Checkpoints directory cleared: {ckpt_dir}\n")
    return initial_epoch
