import argparse
from argparse import Namespace
from typing import Any, Dict, Tuple
import os
import yaml
import torch


def parse_params(dump: bool = False) -> Tuple[Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--model", default="Ours")
    parser.add_argument("--params", default="params/params_fold1.yaml")
    parser.add_argument("--trial_name", default="idxxxx")
    parser.add_argument("--warmup_epochs", default=3, type=int)
    parser.add_argument("--detail_summary", action="store_true")
    parser.add_argument("--imbalance", action="store_true")
    parser.add_argument("--history", default=4, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--calculate_valid_indices",
        action="store_true",
        help="Calculate valid indices for each run",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        required=True,
        help="Select fold number (1-6)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        help="Path to model checkpoint to resume training or testing",
    )
    parser.add_argument(
        "--data_root", default="./datasets", help="Root directory for all datasets"
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="CUDA device index to use"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=1,
        help="Specify the training stage (1 or 2)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "radam_free"],
        default="adamw",
        help="Optimizer to use (adamw or radam_free)",
    )
    parser.add_argument(
        "--cosine_epochs",
        type=int,
        default=20,
        help="Number of epochs for cosine decay",
    )
    args = parser.parse_args()

    # Determine stage from resume_from_checkpoint
    if args.resume_from_checkpoint:
        if "_stage2_" in args.resume_from_checkpoint:
            args.current_stage = 2
            args.imbalance = False
        elif "_stage1_" in args.resume_from_checkpoint:
            args.current_stage = 1
            args.imbalance = True
        else:
            # Use default settings if stage is unknown
            args.current_stage = 1
            args.imbalance = True
    else:
        # Default settings when not resuming from checkpoint
        args.current_stage = 1
        # args.imbalance = True # 勝手に2段階目の学習が始まるのがイヤだったからコメントアウト
    # Device configuration
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        args.device = f"cuda:{args.cuda_device}"
    else:
        args.device = "cpu"

    # Build various paths
    args.data_path = os.path.join(args.data_root, "all_data_hours")
    args.features_path = os.path.join(
        args.data_root, "all_features/completed/all_features_history_672_step_1"
    )
    args.cache_root = os.path.join(args.data_root, "main")

    # Set periods based on fold number
    FOLD_PERIODS = {
        1: {
            "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2019-05-31")],
            "val": [("2011-06-01", "2011-11-30"), ("2019-06-01", "2019-11-30")],
            "test": [("2019-12-01", "2021-11-30")],
        },
        2: {
            "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2019-11-30")],
            "val": [("2011-06-01", "2011-11-30"), ("2019-12-01", "2020-05-31")],
            "test": [("2020-06-01", "2022-05-31")],
        },
        3: {
            "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2020-05-31")],
            "val": [("2011-06-01", "2011-11-30"), ("2020-06-01", "2020-11-30")],
            "test": [("2020-12-01", "2022-11-30")],
        },
        4: {
            "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2020-11-30")],
            "val": [("2011-06-01", "2011-11-30"), ("2020-12-01", "2021-05-31")],
            "test": [("2021-06-01", "2023-05-31")],
        },
        5: {
            "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2021-05-31")],
            "val": [("2011-06-01", "2011-11-30"), ("2021-06-01", "2021-11-30")],
            "test": [("2021-12-01", "2023-11-30")],
        },
        6: {
            "train": [("2011-12-01", "2021-11-30")], #ピッタリ11年間
            "val": [("2011-06-01", "2011-11-30"), ("2021-12-01", "2022-5-30")],
            "test": [("2025-03-01", "2026-3-31")],
        },
    }

    # Select periods for the specified fold
    selected_periods = FOLD_PERIODS[args.fold]
    args.train_periods = selected_periods["train"]
    args.val_periods = selected_periods["val"]
    args.test_periods = selected_periods["test"]

    # Check file extension
    if not args.params.endswith(".yaml"):
        raise ValueError(f"Config file must be a YAML file, got: {args.params}")

    # Load YAML file
    print(f"Loading YAML config from: {args.params}")
    with open(args.params, "r") as f:
        yaml_config = yaml.safe_load(f)

    print("Raw YAML config:")
    print(yaml.dump(yaml_config, default_flow_style=False))

    # Default dataset settings
    dataset_config = {
        "force_preprocess": False,
        "force_recalc_indices": False,
        "force_recalc_stats": False,
    }

    # Merge dataset settings from YAML
    if "dataset" in yaml_config:
        dataset_config.update(yaml_config["dataset"])

    # Process model settings
    model_config = yaml_config.get("model", {})
    model_selected = model_config.get("selected")
    model_models = model_config.get("models", {})

    # Recursively convert model settings to Namespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        else:
            return d

    # Convert model configurations to Namespace objects
    model_models = {
        name: dict_to_namespace(config) for name, config in model_models.items()
    }

    model_namespace = Namespace(selected=model_selected, models=model_models)

    # Merge command line arguments and YAML settings
    args_dict = vars(args)

    # Add top-level settings (weight_decay, lr, epochs, bs, etc.)
    top_level_params = {
        k: v for k, v in yaml_config.items() if k not in ["dataset", "model"]
    }
    args_dict.update(top_level_params)

    # Add dataset and model settings
    args_dict["dataset"] = dataset_config
    args_dict["model"] = model_namespace

    args = Namespace(**args_dict)

    # Set stage2 parameters from YAML
    stage2_config = yaml_config.get("stage2", {})
    args.lr_for_2stage = stage2_config.get("lr")
    args.epoch_for_2stage = stage2_config.get("epochs")

    # Set imbalance based on stage
    if args.mode == "train" and args.resume_from_checkpoint:
        args.imbalance = True if args.stage == 1 else False

    if dump:
        print("\n==========================================")
        print("Final configuration:")
        print(yaml.dump(vars(args), default_flow_style=False))
        print("==========================================")

    return args, yaml_config