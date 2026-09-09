"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/config.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Typed training configuration dataclasses.
----------------------------------------------------------------------------------------------------
    Defines dataclass-based configuration objects (hyper-parameters,
    data, model, training) that drive the end-to-end training pipeline.

"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

__all__ = [
    "EXERCISES",
    "DEFAULT_SPLITS",
    "HyperParameters",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
]

# Gesture class splits (A / B / C / All) per dataset
EXERCISES: dict[str, dict[str, list[int]]] = {
    "DB1": {"A": list(range(1, 13)), "B": list(range(13, 30)), "C": list(range(30, 53)), "All": list(range(1, 53))},
    "DB2": {"A": list(range(1, 18)), "B": list(range(18, 41)), "C": list(range(41, 50)), "All": list(range(1, 50))},
    "DB4": {"A": list(range(1, 13)), "B": list(range(13, 30)), "C": list(range(30, 53)), "All": list(range(1, 53))},
    "DB5": {"A": list(range(1, 13)), "B": list(range(13, 30)), "C": list(range(30, 53)), "All": list(range(1, 53))},
    "Plux": {"All": list(range(1, 11)), "Function": list(range(1, 13))},
}

# Default train / validation repetition indices when splitting by repetition
DEFAULT_SPLITS: dict[str, dict[str, list[int]]] = {
    "DB1": {"train_index": [1, 2, 4, 6, 8, 9, 10], "val_index": [3, 5, 7]},
    "DB2": {"train_index": [1, 3, 4, 6], "val_index": [2, 5]},
    "DB4": {"train_index": [1, 3, 4, 6], "val_index": [2, 5]},
    "DB5": {"train_index": [1, 3, 4, 6], "val_index": [2, 5]},
    "Plux": {"train_index": [1, 2, 4, 6, 8, 9, 10], "val_index": [3, 5, 7]},
}


@dataclass
class HyperParameters:
    """Training hyper-parameters."""

    batch_size: int = 320
    epochs: int = 200
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = None
    patience: int = 100
    lr_patience: int = 5
    learning_rate: float = 1e-3
    lr_factor: float = 0.85
    lr_min: float = 1e-6


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""

    database: str = "DB4"
    source_dir: str = "../Ninapro_DataSet"
    source_type: str = "Envelope"
    source2_type: str = "Feature"
    window_length: int = 200
    exercise: str = "All"
    label_list: list[int] = field(default_factory=lambda: list(range(1, 53)))
    subject_list: list[int] = field(default_factory=lambda: [1])
    channel_list: list[int] = field(default_factory=list)
    process_method: str = "None"
    split_method: str = "Repeat"
    test_split: float = 0.2
    seed: int = 42

    @property
    def source(self) -> str:
        # Directory of the first (envelope) branch data source
        return os.path.join(self.source_dir, self.database, f"{self.database}_{self.source_type}")

    @property
    def source2(self) -> str:
        # Directory of the second (feature) branch data source
        return os.path.join(self.source_dir, self.database, f"{self.database}_{self.source2_type}")

    def resolve_labels(self) -> list[int]:
        # Resolve the effective gesture label list
        if self.exercise:
            return EXERCISES.get(self.database, {}).get(self.exercise, self.label_list)
        return self.label_list

    def default_indices(self) -> tuple[list[int], list[int]]:
        # Return default train / validation repetition indices
        split = DEFAULT_SPLITS.get(self.database, DEFAULT_SPLITS["DB4"])
        return split["train_index"], split["val_index"]


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "DFDFNet"
    fusion_weight: float = 0.5
    fusion_mode: int = 2
    force_fusion: bool = False
    lambda_s_trainable: bool = True
    drb_fusion: bool = True
    ffn_fusion: bool = True
    setcn_branch: bool = True
    irb_branch: bool = True


@dataclass
class TrainConfig:
    """Aggregated training configuration (hyper, data, model)."""

    hyper: HyperParameters = field(default_factory=HyperParameters)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: str = "margin"
    optimizer: str = "adam"
    reload_checkpoint: bool = False
    output_dir: str = "models"
    verbose: int = 1

    @property
    def model_save_dir(self) -> str:
        # Build the model save directory name
        if self.model.fusion_weight == 1:
            source_name = os.path.basename(self.data.source)
        elif self.model.fusion_weight == 0:
            source_name = os.path.basename(self.data.source2)
        else:
            source_name = f"{self.data.database}_Fusion"

        labels = self.data.resolve_labels()
        return (
            f"{self.model.name}-S{self.data.subject_list[0]}-{source_name}"
            f"-sEMG_Percent{int(self.model.fusion_weight * 100)}-{self.data.window_length}ms"
            f"-{len(labels)}-{self.data.split_method}"
        )

    def to_dict(self) -> dict:
        # Serialize the configuration into a dictionary
        return asdict(self)
