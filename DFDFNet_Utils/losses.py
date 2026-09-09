"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/losses.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 21:23:00
@description : 
    Loss functions and loss registry.
----------------------------------------------------------------------------------------------------
    Defines the margin loss and a registry that resolves loss
    callables by name.

"""

from __future__ import annotations
import tensorflow.keras.backend as K
import tensorflow as tf

__all__ = ["margin_loss", "resolve_loss"]


def margin_loss(y_true, y_pred):
    # Margin loss used for capsule-style multi-class training
    lamb, margin = 0.5, 0.1
    positive = y_true * K.square(K.relu(1 - margin - y_pred))
    negative = lamb * (1.0 - y_true) * K.square(K.relu(y_pred - margin))
    result = K.sum(positive + negative, axis=-1)
    return result

_LOSS_REGISTRY = {
    "margin": margin_loss,
    "multiple_margin": margin_loss,
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
    "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(),
    "categorical_focal": tf.keras.losses.CategoricalFocalCrossentropy(),
    "categorical_hinge": tf.keras.losses.CategoricalHinge(),
    "kl_divergence": tf.keras.losses.KLDivergence(),
    "mean_squared_error": tf.keras.losses.MeanSquaredError(),
    "huber": tf.keras.losses.Huber(),
}


def resolve_loss(name: str):
    # Resolve a loss callable by its (case-insensitive) name
    key = (name or "margin").strip().lower()
    if key not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name!r}, available: {sorted(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[key]
