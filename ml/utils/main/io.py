import os
import torch
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, best_valid_gmgs, stage, args):
    checkpoint_dir = os.path.join("checkpoints", "main")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"{args.trial_name}_stage{stage}_best.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_gmgs": best_valid_gmgs,
            "stage": stage,
            "config": vars(args),
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device, scheduler=None):
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except KeyError as e:
        logger.warning(f"Failed to load optimizer state: {e}")
        logger.info("Initializing optimizer with default state")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    if k == "step":
                        state[k] = torch.tensor(0, device=device)

    # Restore scheduler state (if exists)
    if scheduler and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except:
            logger.warning("Failed to load scheduler state")

    best_valid_gmgs = checkpoint.get("best_valid_gmgs", float("-inf"))
    start_epoch = checkpoint.get("epoch", 0) + 1 if "epoch" in checkpoint else 0

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return checkpoint.get("config", {}), best_valid_gmgs, start_epoch
