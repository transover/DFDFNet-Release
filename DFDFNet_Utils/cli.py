"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/cli.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License
@update  : 2026-09-09 16:29:00
@description : 
    Command-line argument parsing.
----------------------------------------------------------------------------------------------------
    Builds the argument parser and dispatches the parsed configuration
    to the end-to-end training pipeline.

"""

from __future__ import annotations

import argparse

from .config import DataConfig, HyperParameters, ModelConfig, TrainConfig
from .pipeline import run
import os,numpy as np, tensorflow as tf

__all__ = ["build_parser", "main"]


def _parse_int_list(value: str) -> list[int]:
    # Parse an integer list string such as ``[1,2,3]``
    return [int(v) for v in value.strip("[]() ").split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    # Build the command-line argument parser
    parser = argparse.ArgumentParser(description="DFDFNet dual-stream dynamic fusion sEMG recognition training program")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("-mn", "--model-name", type=str, default="DFDFNet", help="model name")
    model_group.add_argument("-fw", "--fusion-weight", type=float, default=0.5, help="fusion weight")
    model_group.add_argument("-fm", "--fusion-mode", type=int, default=1, choices=[1, 2, 3], help="fusion mode")
    model_group.add_argument("-fwt", "--lambda-s-trainable", action=argparse.BooleanOptionalAction, default=True, help="whether the fusion weight is trainable")
    model_group.add_argument("--drb-fusion", action=argparse.BooleanOptionalAction, default=True, help="whether to use DRB clustering output")
    model_group.add_argument("--ffn-fusion", action=argparse.BooleanOptionalAction, default=True, help="whether to use FFN output")
    model_group.add_argument("--setcn-branch", action=argparse.BooleanOptionalAction, default=True, help="whether to use the SETCN feature branch")
    model_group.add_argument("--irb-branch", action=argparse.BooleanOptionalAction, default=True, help="whether to use the IRB envelope branch")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("-db", "--database", type=str, default="DB1", help="dataset name")
    data_group.add_argument("-sd", "--source-dir", type=str, default="../Ninapro_DataSet", help="data source root directory")
    data_group.add_argument("-wi", "--window-length", type=int, default=200, help="window length (ms)")
    data_group.add_argument("-ex", "--exercise", type=str, default="All", help="gesture subset A/B/C/All")
    data_group.add_argument("-su", "--subject-list", type=str, default="[1]", help="subject list")
    data_group.add_argument("-ch", "--channel-list", type=str, default="[]", help="channel list")
    data_group.add_argument("-pr", "--process-method", type=str, default="None", help="preprocessing method")
    data_group.add_argument("-tsm", "--train-split-method", type=str, default="Repeat", help="split method Random/Repeat/FoldK")
    data_group.add_argument("-ts", "--test-split", type=float, default=0.2, help="validation ratio for random split")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("-ep", "--epochs", type=int, default=200, help="number of epochs")
    train_group.add_argument("-ba", "--batch-size", type=int, default=320, help="batch size")
    train_group.add_argument("-st", "--steps-per-epoch", type=int, default=None, help="steps per epoch")
    train_group.add_argument("-va", "--validation-steps", type=int, default=None, help="validation steps per epoch")
    train_group.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="initial learning rate")
    train_group.add_argument("-lo", "--loss", type=str, default="margin", help="loss function")
    train_group.add_argument("-irm", "--reload-checkpoint", action="store_true", help="whether to resume from checkpoint")
    train_group.add_argument("-ve", "--verbose", type=int, default=1, help="log verbosity level")
    train_group.add_argument("-o", "--output-dir", type=str, default="models", help="result output directory")
    train_group.add_argument("--seed", type=int, default=42, help="random seed")
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    # Build the training configuration from parsed arguments
    hyper = HyperParameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        learning_rate=args.learning_rate,
    )
    data = DataConfig(
        database=args.database,
        source_dir=args.source_dir,
        window_length=args.window_length,
        exercise=args.exercise,
        subject_list=_parse_int_list(args.subject_list),
        channel_list=_parse_int_list(args.channel_list),
        process_method=args.process_method,
        split_method=args.train_split_method,
        test_split=args.test_split,
        seed=args.seed,
    )
    model = ModelConfig(
        name=args.model_name,
        fusion_weight=args.fusion_weight,
        fusion_mode=args.fusion_mode,
        lambda_s_trainable=args.lambda_s_trainable,
        drb_fusion=args.drb_fusion,
        ffn_fusion=args.ffn_fusion,
        setcn_branch=args.setcn_branch,
        irb_branch=args.irb_branch,
    )
    return TrainConfig(
        hyper=hyper,
        data=data,
        model=model,
        loss=args.loss,
        reload_checkpoint=args.reload_checkpoint,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


def config_os(mode:int=2):
    # Configure the OS/TensorFlow log level (0 all, 1 no info, 2 no warning, 3 no error)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '{}'.format(mode)

def config_seed(seed=1):
    # Configure random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)


def main(argv: list[str] | None = None) -> int:
    # Command-line entry point
    config_os(mode=2)    # set the log level
    config_seed(seed=1)  # configure the random seed

    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
