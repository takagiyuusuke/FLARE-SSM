"""
Train and eval functions used in main.py
"""

from dataclasses import dataclass
import numpy as np
import torch
from utils.main.losses import Losser
from utils.main.statistics import Stat
from tqdm import tqdm
from argparse import Namespace
from typing import Dict, Tuple, Any
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import os
import matplotlib.pyplot as plt


def save_samples(samples, category, trial_name):
    """Function to save samples for qualitative evaluation

    Args:
        samples (list): List of samples to save [(X, true_class, pred_class, file_path), ...]
        category (str): Sample category ("TP" or "TN")
        trial_name (str): Trial name
    """
    base_dir = f"results/qualitative_results/{trial_name}/{category}"
    os.makedirs(base_dir, exist_ok=True)

    for idx, (x, true_class, pred_class, file_path) in enumerate(samples):
        sample_dir = os.path.join(
            base_dir, f"true_{true_class}_pred_{pred_class}_sample{idx}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        # Save images of 12 channels at the last time step
        x_last = x[-1].numpy()  # (12, 256, 256)
        for ch in range(12):
            plt.figure(figsize=(8, 8))
            plt.imshow(x_last[ch], cmap="viridis")
            plt.colorbar()
            plt.title(f"Channel {ch}")
            plt.savefig(os.path.join(sample_dir, f"channel_{ch}.png"))
            plt.close()

        # Save file path information
        with open(os.path.join(sample_dir, "info.txt"), "w") as f:
            f.write(f"File: {file_path}\n")
            f.write(f"True class: {true_class}\n")
            f.write(f"Predicted class: {pred_class}\n")


def train_epoch(
    model: torch.nn.Module,
    optimizer: Adam,
    train_dl: DataLoader,
    losser: Losser,
    stat: Stat,
    args: Namespace,
) -> Tuple[Dict[str, Any], float]:
    """train one epoch"""
    model.train()
    losser.clear()
    stat.clear_all()
    total_loss = 0
    valid_batches = 0
    count = 0

    for X, h, y in tqdm(train_dl):
        # if count > 1:
        #     break
        # count += 1
        X, h, y = X.to(args.device), h.to(args.device), y.to(args.device)
        optimizer.zero_grad()
        output, features = model(X, h)
        loss = losser(output, y, features=features)

        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
            stat.collect(output, y, mode="train")  # Add "valid" here
        else:
            print("Warning: NaN or Inf loss detected. Skipping this batch.")

    if valid_batches == 0:
        print("Error: All batches resulted in NaN or Inf loss. Unable to train.")
        return None, float("inf")

    avg_loss = total_loss / valid_batches
    score = stat.aggregate("train")  # Change to "valid" here too
    return score, avg_loss


def eval_epoch(
    model: torch.nn.Module,
    val_dl: DataLoader,
    losser: Losser,
    stat: Stat,
    args: Namespace,
    mode: str = "valid",
    save_qualitative: bool = False,
    trial_name: str = None,
    optimizer=None,
) -> Tuple[Dict[str, Any], float]:
    """evaluate the given model"""
    assert mode in ["valid", "test"], "Mode must be either 'valid' or 'test'"

    # Set model and optimizer to evaluation mode
    model.eval()
    # Call eval() only for RAdamScheduleFree
    if optimizer is not None and hasattr(optimizer, "eval"):
        optimizer.eval()

    losser.clear()
    stat.clear_all()
    total_loss = 0
    valid_batches = 0
    count = 0

    # For qualitative evaluation results (sequential random sampling)
    if save_qualitative and mode == "test":
        import random

        max_samples_per_class = 20
        tp_samples = []
        tn_samples = []
        tp_prob = (
            max_samples_per_class / 1000
        )  # Sampling probability (adjust as needed)
        tn_prob = (
            max_samples_per_class / 1000
        )  # Sampling probability (adjust as needed)

    with torch.no_grad():
        for batch in tqdm(val_dl):
            # if count > 1:
            #     break
            # count += 1
            if len(batch) == 4:  # If file_paths are included
                X, h, y, file_paths = batch
            else:  # Normal case with 3 elements
                X, h, y = batch
                file_paths = None

            X, h, y = X.to(args.device), h.to(args.device), y.to(args.device)

            output, _ = model(X, h)
            loss = losser(output, y)

            if not torch.isnan(loss) and not torch.isinf(loss):
                stat.collect(output.detach(), y, mode)
                total_loss += loss.item()
                valid_batches += 1

                # Collect samples for qualitative evaluation in test mode (sequential sampling)

                if save_qualitative and mode == "test" and file_paths is not None:
                    pred_classes = torch.argmax(output, dim=1).cpu()
                    true_classes = torch.argmax(y, dim=1).cpu()

                    for i in range(len(pred_classes)):
                        true_class = true_classes[i].item()
                        pred_class = pred_classes[i].item()

                        # 可視化のためのサンプル保存
                        sample_dir = f"results/qualitative_results/{trial_name}/visualization_sample_{i}"
                        os.makedirs(sample_dir, exist_ok=True)

                        # X の最後のタイムステップの各チャンネルを保存
                        x_last = X[i, -1].cpu().numpy()  # (12, 256, 256)
                        for ch in range(x_last.shape[0]):  # 各チャンネルをループ
                            plt.figure(figsize=(8, 8))
                            plt.imshow(x_last[ch], cmap="viridis")
                            plt.colorbar()
                            plt.title(f"Sample {i} - Channel {ch} (True: {true_class}, Pred: {pred_class})")
                            plt.savefig(os.path.join(sample_dir, f"channel_{ch}.png"))
                            plt.close()

                        # ファイルパス情報を保存
                        with open(os.path.join(sample_dir, "info.txt"), "w") as f:
                            f.write(f"File: {file_paths[i]}\n")
                            f.write(f"True class: {true_class}\n")
                            f.write(f"Predicted class: {pred_class}\n")

                        # If TP condition is met and probabilistically selected
                        if (
                            true_class < 2
                            and pred_class < 2
                            and true_class == pred_class
                            and len(tp_samples) < max_samples_per_class
                            and random.random() < tp_prob
                        ):
                            tp_samples.append(
                                (X[i].cpu(), true_class, pred_class, file_paths[i])
                            )

                        # If TN condition is met and probabilistically selected
                        elif (
                            true_class >= 2
                            and pred_class >= 2
                            and true_class == pred_class
                            and len(tn_samples) < max_samples_per_class
                            and random.random() < tn_prob
                        ):
                            tn_samples.append(
                                (X[i].cpu(), true_class, pred_class, file_paths[i])
                            )

                        # Exit if enough samples have been collected
                        if (
                            len(tp_samples) >= max_samples_per_class
                            and len(tn_samples) >= max_samples_per_class
                        ):
                            break

            del output
            torch.cuda.empty_cache()

    # Save qualitative evaluation results
    if save_qualitative and mode == "test":
        if tp_samples:
            save_samples(tp_samples, "TP", trial_name)
        if tn_samples:
            save_samples(tn_samples, "TN", trial_name)

        # Explicitly free memory
        del tp_samples
        del tn_samples
        torch.cuda.empty_cache()

    avg_loss = total_loss / valid_batches
    score = stat.aggregate(mode)
    return score, avg_loss
