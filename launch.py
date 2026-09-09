"""
@project : DFDFNet-Release
@file    : train.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 21:16:00
@description : 
    Top-level training entry point.
----------------------------------------------------------------------------------------------------
    Launches the DFDFNet training program by delegating to the
    command-line interface entry point in DFDFNet_Utils.cli.

Usage::

    python train.py --database DB2 --exercise All --subject-list "[1]" --epochs 200
    python train.py -db DB2 -ex A -la 0.5 -fm 2 -ba 320 -ep 200
"""

from DFDFNet_Utils.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
