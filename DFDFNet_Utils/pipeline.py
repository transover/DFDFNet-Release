"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/pipeline.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    End-to-end training pipeline orchestration.
----------------------------------------------------------------------------------------------------
    Orchestrates the full training workflow: data loading, splitting,
    model building, training, evaluation, and result saving.

"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from .config import DataConfig, TrainConfig
from .data import load_nina_pro, random_split, split_dataset
from .model import build_model_DFDFNet
from .training import Trainer
from .utils import (
    configure_console_encoding,
    format_duration,
    print_device_summary,
    print_environment,
    set_seed,
    set_tf_log_level,
)

__all__ = ["run"]


def _print_section(title: str) -> None:
    # Print a section separator title
    print("\n" + "=" * 24 + f" {title} " + "=" * 24)


def _print_dataset_info(data: DataConfig) -> None:
    # Print dataset configuration information
    _print_section("Dataset Sample Config")
    print(f"Subject list: {data.subject_list}")
    print(f"Channel selection: {data.channel_list}")
    print(f"Gesture label list: {data.resolve_labels()}")
    print(f"Preprocessing method: {data.process_method}")
    print(f"Split method: {data.split_method}")


def _model_inputs(
    model_cfg,
    data: Any,
    data2: Any,
) -> Any:
    # Return the actual model inputs based on the model configuration
    if model_cfg.fusion_weight == 1 and not model_cfg.force_fusion:
        return data
    if model_cfg.fusion_weight == 0 and not model_cfg.force_fusion:
        return data2
    return [data, data2]


def _split_data(
    data: DataConfig,
    x: np.ndarray,
    y: np.ndarray,
    x2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Split train / validation sets according to the configured method
    if data.split_method == "Random":
        train_x, val_x, train_y, val_y = random_split(x, y, data.test_split, data.seed)
        train_x2, val_x2, _, _ = random_split(x2, y, data.test_split, data.seed)
    elif data.split_method == "Repeat":
        train_index, val_index = data.default_indices()
        train_x, val_x, train_y, val_y = split_dataset(x, y, val_index=val_index, train_index=train_index)
        train_x2, val_x2, _, _ = split_dataset(x2, y, val_index=val_index, train_index=train_index)
    elif "Fold" in data.split_method:
        k = int(data.split_method[-1])
        index_list = list(range(1, 11))
        val_index = random.sample(index_list, k)
        train_index = [i for i in index_list if i not in val_index]
        print(f"10-Fold dataset split: train {train_index}   val {val_index}")
        train_x, val_x, train_y, val_y = split_dataset(x, y, val_index=val_index, train_index=train_index)
        train_x2, val_x2, _, _ = split_dataset(x2, y, val_index=val_index, train_index=train_index)
    else:
        raise ValueError(f"Unknown split method: {data.split_method!r}")
    return train_x, val_x, train_y, val_y, train_x2, val_x2


def run(config: TrainConfig) -> dict[str, Any]:
    # Run the full training and evaluation pipeline
    configure_console_encoding()
    set_tf_log_level(2)
    set_seed(config.data.seed)
    print_environment()

    data, model_cfg = config.data, config.model
    labels = data.resolve_labels()
    _print_dataset_info(data)

    # 1. Extract MVC normalization extrema
    _print_section("Preprocessing and Data Loading")
    mvc = load_nina_pro(
        source=data.source,
        label_list=labels,
        subject_list=data.subject_list,
        channel_list=data.channel_list,
        window_length=data.window_length,
        process_method="None",
        is_mvc=True,
        return_mvc=True,
    )
    # 2. Load the first branch (envelope) data
    x, y = load_nina_pro(
        source=data.source,
        label_list=labels,
        subject_list=data.subject_list,
        channel_list=data.channel_list,
        window_length=data.window_length,
        process_method=data.process_method,
        is_mvc=True,
        mvc_max=mvc["max"],
        mvc_min=mvc["min"],
    )
    # 3. Load the second branch (feature) data
    x2, y2 = load_nina_pro(
        source=data.source2,
        label_list=labels,
        subject_list=data.subject_list,
        channel_list=data.channel_list,
        window_length=data.window_length,
        process_method=data.process_method,
    )

    # 4. Split train / validation sets
    train_x, val_x, train_y, val_y, train_x2, val_x2 = _split_data(data, x, y, x2)
    print(f"Train size: {train_x.shape[0]}   Val size: {val_x.shape[0]}")

    # 5. Build the model
    _print_section("Model Design Info")
    model = build_model_DFDFNet(
        input_shape=x.shape[1:],
        feature_input_shape=x2.shape[1:],
        num_classes=int(y.shape[-1]),
        fusion_weight=model_cfg.fusion_weight,
        fusion_mode=model_cfg.fusion_mode,
        force_fusion=model_cfg.force_fusion,
        name=config.model_save_dir,
        lambda_s_trainable=model_cfg.lambda_s_trainable,
        drb_fusion=model_cfg.drb_fusion,
        ffn_fusion=model_cfg.ffn_fusion,
        setcn_branch=model_cfg.setcn_branch,
        irb_branch=model_cfg.irb_branch,
    )

    # 6. Train
    train_inputs = _model_inputs(model_cfg, train_x, train_x2)
    val_inputs = _model_inputs(model_cfg, val_x, val_x2)
    trainer = Trainer(config)
    trainer.compile(model)
    train_start = time.perf_counter()
    history = trainer.fit(
        model,
        train_inputs,
        train_y,
        val_inputs,
        val_y,
    )
    train_elapsed = time.perf_counter() - train_start

    # 7. Evaluate and save
    _print_section("Model Evaluation")
    result = trainer.evaluate(model, val_inputs, val_y)
    trainer.save_model(model)
    trainer.save_history(history)
    trainer.save_config()
    trainer.save_report(model, result, labels)

    # 8. Summarize training resources and elapsed time
    print_device_summary()
    print(f"➣ Total training time: {format_duration(train_elapsed)}")

    print(f"\n➣ Global accuracy: {result['accuracy']:.4f}   Global loss: {result['loss']:.4f}")
    print(f"➣ Macro F1: {np.mean(result['f1_per_class']):.4f}")
    print(f"\n➣ {config.model_save_dir} training finished!")
    return result
