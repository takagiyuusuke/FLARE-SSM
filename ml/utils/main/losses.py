import torch
import numpy as np

from dataclasses import dataclass
from torch import nn
from torch import Tensor

CLASS_SAMPLES = {
    1: torch.tensor([9985, 11441, 8051, 1608]),  # fold1
    2: torch.tensor([12083, 11441, 8051, 1608]),  # fold2
    3: torch.tensor([14170, 11441, 8084, 1608]),   # fold3
    6: torch.tensor([18078, 13377, 8469, 1675])   # fold6
}

@dataclass
class LossConfig:
    lambda_bss: float
    lambda_gmgs: float
    lambda_ce: float
    score_mtx: torch.Tensor
    fold: int
    class_weights: dict
    model_name: str
    stage: int 

def calculate_weights(config: dict, samples: torch.Tensor) -> torch.Tensor:
    """Calculate weights"""
    if not config["enabled"]:
        return None
        
    method = config["method"]
    if method == "none":
        return None
    elif method == "custom":
        weights = torch.tensor(config["custom_weights"], dtype=torch.float)
    elif method == "inverse":
        weights = 1.0 / samples.float()
    elif method == "effective_samples":
        beta = config.get("beta", 0.9999)
        effective_num = 1.0 - torch.pow(beta, samples.float())
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * len(samples)
    return weights

class Losser:
    def __init__(self, config: LossConfig, device: str):
        self.config = config
        self.device = device
        self.accum = []

        # Get weight settings for the current stage
        stage_key = f"stage{self.config.stage}"
        stage_weights_config = self.config.class_weights[stage_key]

        # Calculate class weights
        weights = None
        if stage_weights_config["enabled"]:
            samples = CLASS_SAMPLES[self.config.fold]
            weights = calculate_weights(
                stage_weights_config,
                samples)
            if weights is not None:
                weights = weights.to(device)
        self.weights = weights

        # self.ce_loss = nn.CrossEntropyLoss(weight=weights).to(device)
        self.alpha = weights
        self.gamma = 1.0

        # Cross Entropy Loss with weighting
        self.ce_loss = nn.CrossEntropyLoss(weight=weights).to(device)

    def __call__(self, y_pred: Tensor, y_true: Tensor, features: Tensor = None) -> Tensor:
        """
        Compute loss
        """
        # Additional losses only for Ours model
        if self.config.model_name == "Ours":
            # Cross Entropy Loss with weighting
            if features is not None:
                ce = self.calc_ib_loss(y_pred, y_true, features)
            else:
                ce = self.focal_loss(y_pred, y_true, self.alpha, self.gamma)
            loss = self.config.lambda_ce * ce
            # GMGS loss
            if self.config.lambda_gmgs > 0:
                gmgs_loss = self.calc_gmgs_loss(y_pred, y_true)
                if not gmgs_loss.isnan():
                    loss = loss + self.config.lambda_gmgs * gmgs_loss
                
            # BSS loss
            if self.config.lambda_bss > 0:
                bss_loss = self.calc_bss_loss(y_pred, y_true)
                if not bss_loss.isnan():
                    loss = loss + self.config.lambda_bss * bss_loss
        else:
            # Cross Entropy Loss with weighting
            ce = self.ce_loss(y_pred, torch.argmax(y_true, dim=1))
            # ce = self.focal_loss(y_pred, y_true, self.alpha, self.gamma)
            loss = self.config.lambda_ce * ce

        self.accum.append(loss.clone().detach().cpu().item())
        return loss

    def calc_gmgs_loss(self, y_pred: Tensor, y_true) -> Tensor:
        """
        Compute GMGS loss
        """
        score_mtx = torch.tensor(self.config.score_mtx).to(self.device)
        y_truel = torch.argmax(y_true, dim=1)
        weight = score_mtx[y_truel]
        py = torch.log(y_pred)
        output = torch.mul(y_true, py)
        output = torch.mul(output, weight)
        output = torch.mean(output)
        return -output

    def calc_bss_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute BSS loss with class weighting
        """
        tmp = y_pred - y_true
        tmp = torch.mul(tmp, tmp)
        tmp = torch.sum(tmp, dim=1)

        # Apply class weights based on inverse sample counts
        if self.weights is not None:
            sample_weights = self.weights[torch.argmax(y_true, dim=1)]
        else:
            sample_weights = torch.ones_like(tmp)

        tmp = torch.mul(tmp, sample_weights)
        tmp = torch.mean(tmp)
        return tmp

    def get_mean_loss(self) -> float:
        """
        Get mean loss
        """
        return np.mean(self.accum)
    
    def focal_loss(self, y_pred, y_true, alpha, gamma=1.0):
        """
        y_pred: [B, C] (logits, softmax前)
        y_true: [B, C] (one-hot)
        alpha: [C] (class weights)
        """
        ce = torch.nn.functional.cross_entropy(y_pred, torch.argmax(y_true, dim=1), weight=alpha, reduction='none')
        pt = torch.exp(-ce)  # pt = softmax prob of correct class
        focal = (1 - pt) ** gamma * ce
        return focal.mean()
    
    def calc_ib_loss(self, y_pred: Tensor, y_true: Tensor, features: Tensor) -> Tensor:
        """
        Compute Influence-Balanced Loss (式(6))
        y_pred: [B, C] (logits)
        y_true: [B, C] (one-hot)
        features: [B, F] (FC層直前の特徴量)
        """
        features = features.mean(dim=1)
        if y_true.ndim == 2:
            y_true_indices = torch.argmax(y_true, dim=1)
        else:
            y_true_indices = y_true

        # サンプルごとのクロスエントロピー
        ce = torch.nn.functional.cross_entropy(y_pred, y_true_indices, reduction='none')  # [B]

        # softmax 出力と one-hot との差分のL1ノルム
        probs = torch.softmax(y_pred, dim=1)
        y_onehot = torch.nn.functional.one_hot(y_true_indices, num_classes=y_pred.size(1)).float().to(y_pred.device)
        delta = torch.abs(probs - y_onehot).sum(dim=1)  # [B]
        feature_norm = torch.abs(features).sum(dim=1)   # [B]

        # λ_k
        if self.weights is not None:
            lambda_k = self.weights[y_true_indices]  # [B]
        else:
            lambda_k = torch.ones_like(delta)

        # IB loss
        eps = 1e-3
        numerator = lambda_k * ce
        denominator = delta * feature_norm + eps
        loss_ib = numerator / denominator
        return loss_ib.mean()

    def clear(self):
        """
        Clear accumulated loss
        """
        self.accum.clear()
