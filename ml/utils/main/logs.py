import wandb as wandb_runner
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import logging
import numpy as np
from torchinfo import summary
import csv


@dataclass
class Log:
    stage: str
    loss: float
    score: Any


class Logger:
    def __init__(self, args: Namespace, wandb: bool) -> None:
        if args.wandb:
            wandb_runner.init(project="SolarFlarePrediction2", name=args.trial_name)
        self.wandb_enabled = wandb
        self.logger = setup_logging(args.trial_name)

    def write(self, epoch: int, logs: List[Log]):
        l: Dict[str, Any] = {"epoch": epoch}
        for lg in logs:
            l[f"{lg.stage}_loss"] = lg.loss
            l.update(lg.score)

        if self.wandb_enabled:
            wandb_runner.log(l)

    def print_model_summary(self, model, args=None, mock_sample=None):
        """Print model summary"""
        if args and args.detail_summary and mock_sample:
            summary(model, [(args.bs, *feature.shape) for feature in mock_sample[0]])
        else:
            summary(model)

    def print_best_metrics(self, experiment, stage: str = "1st"):
        """Output evaluation metrics for the best model"""
        self.logger.info(f"\n========== Best Model Metrics ({stage} stage) ==========")
        self.logger.info(f"Best Valid GMGS: {experiment.best_valid_gmgs:.4f}")
        self.logger.info(f"Train Loss: {experiment.best_train_loss:.4f}")
        self.logger.info(f"Valid Loss: {experiment.best_valid_loss:.4f}")

        # Output scores in specified order
        metrics = ["GMGS", "BSS", "TSS", "ACC"]

        # Valid scores
        valid_scores = [experiment.best_valid_score[f"valid_{m}"] for m in metrics]
        self.logger.info(
            f"Valid Scores: {' '.join(f'{m}: {s:.4f}' for m, s in zip(metrics, valid_scores))}"
        )

        # Test scores
        test_scores = [experiment.test_score[f"test_{m}"] for m in metrics]
        self.logger.info(
            f"Test Scores: {' '.join(f'{m}: {s:.4f}' for m, s in zip(metrics, test_scores))}"
        )

        # Display confusion matrices
        self.logger.info("\nConfusion Matrices:")

        # Valid confusion matrix
        self.logger.info("Valid Confusion Matrix:")
        valid_cm = experiment.stat.confusion_matrix(
            np.array(experiment.best_valid_predictions),
            np.array(experiment.best_valid_observations),
        )
        for row in valid_cm:
            self.logger.info(f"    {row}")

        # Test confusion matrix
        self.logger.info("\nTest Confusion Matrix:")
        test_cm = experiment.stat.confusion_matrix(
            np.array(experiment.test_predictions),
            np.array(experiment.test_observations),
        )
        for row in test_cm:
            self.logger.info(f"    {row}")

        self.logger.info("=" * 50)

        # Save metrics to CSV
        self._save_metrics_to_csv(
            experiment.args.trial_name,
            stage,
            experiment.best_train_loss,
            experiment.best_valid_loss,
            valid_scores,
            test_scores,
            valid_cm,
            test_cm,
        )

    def _save_metrics_to_csv(
        self,
        trial_name: str,
        stage: str,
        train_loss: float,
        valid_loss: float,
        valid_scores: List[float],
        test_scores: List[float],
        valid_cm: np.ndarray,
        test_cm: np.ndarray,
    ):
        """Save metrics to CSV file"""
        csv_dir = os.path.join("logs", "main", "csvs")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{trial_name}_stage{stage}.csv")

        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(
                [
                    "train_loss",
                    "valid_loss",
                    "valid_GMGS",
                    "valid_BSS-M",
                    "valid_TSS-M",
                    "valid_ACC",
                    "test_GMGS",
                    "test_BSS-M",
                    "test_TSS-M",
                    "test_ACC",
                ]
            )

            # Write scores
            scores = [f"{train_loss:.4f}", f"{valid_loss:.4f}"]
            scores.extend([f"{score:.4f}" for score in valid_scores])
            scores.extend([f"{score:.4f}" for score in test_scores])
            writer.writerow(scores)
            f.write("\n")

            # Write confusion matrices
            f.write("Valid Confusion Matrix:\n")
            for row in valid_cm:
                f.write(f"[{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]\n")

            f.write("\nTest Confusion Matrix:\n")
            for row in test_cm:
                f.write(f"[{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}]\n")

        self.logger.info(f"Metrics saved to {csv_path}")


def setup_logging(trial_name):
    log_dir = os.path.join("logs", "main")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{trial_name}.log")

    # 既存のロガーがあれば取得、なければ新規作成
    logger = logging.getLogger(trial_name)
    
    # ハンドラが既に設定されている場合は追加しない
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    # 親ロガーからの伝播を防止
    logger.propagate = False

    return logger
