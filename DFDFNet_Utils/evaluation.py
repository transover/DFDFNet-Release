"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/evaluation.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Model evaluation and metric computation.
----------------------------------------------------------------------------------------------------
    Evaluates the model and computes accuracy, F1, confusion matrix,
    and macro-averaged ROC-AUC metrics.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, roc_curve

__all__ = ["evaluate_model", "macro_roc", "classification_report_df"]


def evaluate_model(
    model,
    x: Any,
    y: np.ndarray,
    batch_size: int = 320,
) -> dict[str, Any]:
    # Evaluate the model and compute classification metrics
    loss, accuracy = model.evaluate(x, y, batch_size=batch_size, verbose=0)
    y_pred = model.predict(x, batch_size=batch_size, verbose=0)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y, axis=1)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "y_true": y,
        "y_pred": y_pred,
        "y_true_label": y_true_label,
        "y_pred_label": y_pred_label,
        "f1_per_class": f1_score(y_true_label, y_pred_label, average=None),
        "confusion_matrix": confusion_matrix(y_true_label, y_pred_label),
    }


def macro_roc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    # Compute macro-averaged ROC curves and AUC
    num_classes = y_true.shape[1]
    fpr, tpr, aucs = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        aucs[i] = auc(fpr[i], tpr[i])

    fpr_all = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    tpr_all = np.zeros_like(fpr_all)
    for i in range(num_classes):
        tpr_all += np.interp(fpr_all, fpr[i], tpr[i])
    macro_tpr = tpr_all / num_classes

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": aucs,
        "macro_fpr": fpr_all,
        "macro_tpr": macro_tpr,
        "macro_auc": float(auc(fpr_all, macro_tpr)),
    }


def classification_report_df(y_true_label: np.ndarray, y_pred_label: np.ndarray, num_classes: int) -> pd.DataFrame:
    # Generate a classification report as a DataFrame
    report = classification_report(
        y_true_label,
        y_pred_label,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report)
