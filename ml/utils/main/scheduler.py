import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=5, gamma=0.95, last_epoch=-1
    ):
        """
        Args:
            optimizer (Optimizer): Optimization algorithm
            T_0 (int): Length of the first cycle (number of epochs)
            T_mult (int, optional): Cycle multiplier (default: 1)
            eta_max (float, optional): Maximum learning rate (default: 0.1)
            T_up (int, optional): Warm-up period (number of epochs, default: 5)
            gamma (float, optional): Learning rate decay rate (default: 0.95)
            last_epoch (int, optional): Last epoch (default: -1)
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_up = T_up
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            # Warm-up period: linearly increase learning rate
            return [
                base_lr + (self.eta_max - base_lr) * (self.T_cur / self.T_up)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine Annealing with Warm Restarts
            T_i = self.T_0 * (self.T_mult**self.cycle)
            t = self.T_cur - self.T_up
            return [
                self.eta_max
                * (self.gamma**self.cycle)
                * (1 + math.cos(math.pi * t / T_i))
                / 2
                for _ in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_0 * (self.T_mult**self.cycle):
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_0 * (self.T_mult ** (self.cycle - 1))
        else:
            if epoch < 0:
                raise ValueError("Epoch must be a non-negative integer.")
            self.T_cur = epoch
            if self.T_cur >= self.T_0 * (self.T_mult**self.cycle):
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_0 * (self.T_mult ** (self.cycle - 1))

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def get_scheduler(optimizer, scheduler_config: Optional[dict] = None):
    """
    If scheduler_config type is "none", the learning rate will be fixed.
    """
    if scheduler_config is None or scheduler_config.get("type") == "none":
        return None

    scheduler_type = scheduler_config.get("type")
    scheduler_params = scheduler_config.get("params", {})

    if scheduler_type == "cosine_annealing_warm_restarts":
        return CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=scheduler_params.get("T_0", 25),
            T_mult=scheduler_params.get("T_mult", 1),
            eta_max=scheduler_params.get("eta_max", 7.5e-5),
            T_up=scheduler_params.get("T_up", 5),
            gamma=scheduler_params.get("gamma", 0.95),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
