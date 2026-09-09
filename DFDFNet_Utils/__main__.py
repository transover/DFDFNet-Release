"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/__main__.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Module-level execution entry point.
----------------------------------------------------------------------------------------------------
    Enables launching the training program via ``python -m DFDFNet_Utils``.

"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
