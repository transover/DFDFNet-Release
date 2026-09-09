"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/model.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 21:23:00
@description : 
    DFDFNet model construction.
----------------------------------------------------------------------------------------------------
    Builds the dual-stream dynamic fusion network by composing the
    sEMG envelope branch, the feature branch, and the decision-level
    fusion head.

"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .blocks import (
    DecisionScoreHead_FeatureNet,
    DecisionScoreHead_sEMGNet,
    InvertedResidualBlock,
    LearnableFusion,
    LearnableWeightedSum,
    SqueezeExcitation,
    TemporalConvBlock,
    FeedForwardRefBlock,
    FeedForwardFeaBlock
)

__all__ = ["build_dfdfnet"]


def _semg_branch(inputs: keras.KerasTensor, name: str = "semg_net", IRB_flag: bool = True, dropout = 0.0) -> keras.KerasTensor:
    # sEMG envelope branch (sEMGNet) built with an inverted residual block
    channels = int(inputs.shape[-1])
    if IRB_flag:
        print('Using InvertedResidualBlock for sEMGNet')
        x = InvertedResidualBlock(
            expanded_channels=60,
            output_channels=channels,
            strides=1,
            kernel_size=6,
            is_use_expand_conv = True,
            name=f"{name}_irb",
        )(inputs)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return x

def _feature_branch(inputs: keras.KerasTensor, name: str = "feature_net", SETCN_flag: bool = True) -> keras.KerasTensor:
    # sEMG feature branch (FeatureNet) built with squeeze-excitation and temporal conv
    channels = int(inputs.shape[-1])
    if SETCN_flag:
        print('Using SqueezeExcitation for FeatureNet')
        x = SqueezeExcitation(ratio=2, name=f"{name}_se")(inputs)
        x = TemporalConvBlock(
            filters=(channels,),
            kernel_size=3,
            pool_size=1,
            norm="sn",
            activation="relu",
            name=f"{name}_tcnn",
        )(x)
    return x


def build_model_DFDFNet(
    input_shape: tuple[int, int],
    feature_input_shape: tuple[int, int],
    num_classes: int,
    fusion_weight: float = 0.5,
    fusion_mode: int = 2,
    force_fusion: bool = False,
    name: str = "DFDFNet",
    **kwargs,
) -> keras.Model:
    # Build the DFDFNet dual-stream dynamic fusion network
    lambda_s_trainable_flag = kwargs.get("lambda_s_trainable", True)  # whether lambda_s is trainable
    DRBFusion_flag = kwargs.get("drb_fusion", True)  # whether to use DRB clustering output
    FFNFusion_flag = kwargs.get("ffn_fusion", True)  # whether to use FFN output
    SETCN_flag = kwargs.get("setcn_branch", True)  # whether to use the SETCN feature branch
    IRB_flag = kwargs.get("irb_branch", True)  # whether to use the IRB envelope branch

    # Input layers
    input_RefData = layers.Input(shape=input_shape, name="input_semg_envelope")
    input_FeaData = layers.Input(shape=feature_input_shape, name="input_semg_feature")

    # Feature fusion
    if fusion_weight == 0 and not force_fusion:
        output_FeaData = _feature_branch(input_FeaData, name="feature_net", SETCN_flag=SETCN_flag)
        print('sEMG feature branch output shape: {}'.format(output_FeaData.shape))
        inputs = input_FeaData
        outputs = FeedForwardFeaBlock(num_classes, use_hidden=True, name="head_semg")(output_FeaData)       # FFN output mapping

    elif fusion_weight == 1 and not force_fusion:
        output_RefData = _semg_branch(input_RefData, name="semg_net", IRB_flag=IRB_flag)
        print('sEMG envelope branch output shape: {}'.format(output_RefData.shape))
        inputs = input_RefData
        outputs = FeedForwardRefBlock(num_classes, use_hidden=True, name="head_semg")(output_RefData)       # FFN output mapping
    else:
        # Dual-branch decision-level fusion (default)
        print('\033[1;35;10mλ={} Fusion learning (FusionNet) m={} fusion mode: decision fusion\033[0;0;0m'.format(fusion_weight, fusion_mode))
        output_RefData = _semg_branch(input_RefData, name="semg_net", IRB_flag=IRB_flag)
        output_FeaData = _feature_branch(input_FeaData, name="feature_net", SETCN_flag=SETCN_flag)

        if FFNFusion_flag:
            print('Using DecisionScoreHead_sEMGNet for sEMG branch decision mapping')
            strategic_decision_score_semg= DecisionScoreHead_sEMGNet(num_classes, use_hidden=True, name="decision_space_semg")(output_RefData)       # FFN output mapping
        if DRBFusion_flag:
            print('Using DecisionScoreHead_FeatureNet for feature branch decision mapping')
            strategic_decision_score_feat = DecisionScoreHead_FeatureNet(num_classes, name="decision_space_feat")(output_FeaData)
        if lambda_s_trainable_flag:
            print('Using LearnableWeightedSum for decision-level fusion')
            strategic_decision_score_weighted_sum = LearnableWeightedSum(name="decision_fusion")([strategic_decision_score_semg, strategic_decision_score_feat])

        inputs = [input_RefData, input_FeaData]
        outputs = layers.Dense(num_classes, activation="softmax")(strategic_decision_score_weighted_sum)    # softmax normalization to class probabilities

    # Build the global model
    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    return model
