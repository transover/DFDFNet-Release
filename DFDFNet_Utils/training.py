"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/training.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Trainer for model compilation, training and evaluation.
----------------------------------------------------------------------------------------------------
    Implements the Trainer class that handles model compilation,
    checkpoint resume, training, evaluation, and result persistence.

"""

from __future__ import annotations

import json
import os
from contextlib import redirect_stdout
from typing import Any, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from .callbacks import build_callbacks, prepare_checkpoint
from .config import TrainConfig
from .evaluation import classification_report_df, evaluate_model, macro_roc
from .losses import resolve_loss
from .utils import ensure_dir

__all__ = ["Trainer"]


class Trainer:
    """DFDFNet trainer handling compilation, resume, training, evaluation and saving."""

    def __init__(self, config: TrainConfig):
        # Initialize the trainer with the full training configuration
        self.config = config
        self.hp = config.hyper

    @property
    def model_dir(self) -> str:
        # Directory for the current model's outputs
        return os.path.join(self.config.output_dir, self.config.model_save_dir)

    def compile(self, model: keras.Model) -> None:
        # Compile the model with loss, optimizer and metrics
        loss = resolve_loss(self.config.loss)
        optimizer = keras.optimizers.Adam(learning_rate=self.hp.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    def save_structure(self, model: keras.Model) -> None:
        # Print and save the model structure summary
        model.summary()
        structure_dir = ensure_dir(os.path.join(self.model_dir, "structure"))
        with open(os.path.join(structure_dir, "model.txt"), "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                model.summary()

    def fit(
        self,
        model: keras.Model,
        train_x: Any,
        train_y: np.ndarray,
        val_x: Any,
        val_y: np.ndarray,
    ):
        # Train the model and return the training history
        self.save_structure(model)
        initial_epoch = prepare_checkpoint(
            model,
            self.config.reload_checkpoint,
            self.config.model_save_dir,
            self.config.output_dir,
        )
        callbacks = build_callbacks(self.hp, self.config.model_save_dir, self.config.output_dir)

        history = model.fit(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            batch_size=self.hp.batch_size,
            epochs=self.hp.epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=self.hp.steps_per_epoch,
            validation_steps=self.hp.validation_steps,
            callbacks=callbacks,
            shuffle=True,
            verbose=self.config.verbose,
        )
        return history

    def evaluate(self, model: keras.Model, val_x: Any, val_y: np.ndarray) -> dict:
        # Evaluate the model and return the metrics dictionary
        return evaluate_model(model, val_x, val_y, self.hp.batch_size)

    def save_model(self, model: keras.Model) -> None:
        # Save the complete model
        save_dir = ensure_dir(os.path.join(self.model_dir, "model"))
        model.save(os.path.join(save_dir, "model.keras"))
        print(f"➣ Model saved to {save_dir}")

    def save_history(self, history) -> pd.DataFrame:
        # Save the training history as a CSV DataFrame
        df = pd.DataFrame(history.history)
        df["epoch"] = history.epoch
        history_dir = ensure_dir(os.path.join(self.model_dir, "history"))
        df.to_csv(os.path.join(history_dir, "model.csv"), index=False, encoding="utf-8")
        return df

    def save_config(self) -> None:
        # Save the training configuration as JSON
        info_dir = ensure_dir(os.path.join(self.model_dir, "info"))
        with open(os.path.join(info_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=4)

    def save_report(
        self,
        model: keras.Model,
        result: dict,
        labels: list[int],
    ) -> None:
        # Save the classification report, confusion matrix and ROC-AUC metrics
        result_dir = ensure_dir(os.path.join(self.model_dir, "result"))

        report = classification_report_df(result["y_true_label"], result["y_pred_label"], len(labels))
        report.to_csv(os.path.join(result_dir, "precision_recall.csv"), encoding="utf-8")

        np.savetxt(
            os.path.join(result_dir, "confusion_matrix.csv"),
            result["confusion_matrix"],
            delimiter=",",
        )

        roc = macro_roc(result["y_true"], result["y_pred"])
        roc_df = pd.DataFrame(
            {
                "macro_fpr": roc["macro_fpr"],
                "macro_tpr": roc["macro_tpr"],
            }
        )
        roc_df.to_csv(os.path.join(result_dir, "ROC_AUC.csv"), index=False, encoding="utf-8")

        summary = {
            "model": model.name,
            "params": model.count_params(),
            "accuracy": result["accuracy"],
            "loss": result["loss"],
            "macro_f1": float(np.mean(result["f1_per_class"])),
            "macro_auc": roc["macro_auc"],
        }
        with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        print(f"➣ Evaluation results saved to {result_dir}")
