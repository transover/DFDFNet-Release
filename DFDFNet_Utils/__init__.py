"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/__init__.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Package initialization and public API exports.
----------------------------------------------------------------------------------------------------
    Exposes the public API of DFDFNet_Utils, including configuration
    classes, the model builder, and the Trainer.

"""

from .config import DataConfig, HyperParameters, ModelConfig, TrainConfig
from .model import build_model_DFDFNet
from .training import Trainer

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "HyperParameters",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "build_model_DFDFNet",
    "Trainer",
]
