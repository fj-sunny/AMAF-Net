"""
损失函数。

FocalLoss 与 ProbabilisticContrastiveLoss 按 train_missing_aware_fusion_proco.py 实现。
ProCo 把融合特征当作高斯分布，用类别原型距离做对比损失。
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)。"""

    def __init__(
        self,
        alpha: Optional[Union[float, int, Sequence[float]]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            n = num_classes if num_classes is not None else 1
            self.register_buffer("alpha", torch.tensor([float(alpha)] * n))
        else:
            self.register_buffer("alpha", torch.tensor(list(alpha), dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha = self.alpha
            if alpha.numel() == 1 and self.num_classes:
                alpha = alpha.repeat(self.num_classes)
            focal_loss = alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ProbabilisticContrastiveLoss(nn.Module):
    """
    概率对比学习（ProCo）。

    μ_i = f_i，Σ_i = diag(exp(ψ(f_i)))。
    D(z_i, c) = ||μ_i - μ_c||^2 + λ_u * Tr(Σ_i + Σ_c)。
    L = -log softmax(-D / τ) 在真实类别上的取值。
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        temperature: float = 0.07,
        lambda_uncertainty: float = 0.1,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.lambda_uncertainty = lambda_uncertainty
        self.momentum = momentum
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim),
        )
        self.register_buffer("prototype_mean", torch.zeros(num_classes, feat_dim))
        self.register_buffer("prototype_cov", torch.zeros(num_classes, feat_dim))
        self.register_buffer("prototype_count", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("running_mean", torch.zeros(num_classes, feat_dim))
        self.register_buffer("running_cov", torch.zeros(num_classes, feat_dim))
        self.register_buffer("running_count", torch.zeros(num_classes, dtype=torch.long))

    def reset_prototypes(self):
        self.running_mean.zero_()
        self.running_cov.zero_()
        self.running_count.zero_()

    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            log_sigma_diag = self.uncertainty_head(features.detach())
            sigma_diag = torch.exp(log_sigma_diag.clamp(max=10.0))
            mu_i = features.detach()
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                mu_c_batch = mu_i[mask].mean(dim=0)
                diff = mu_i[mask] - mu_c_batch.unsqueeze(0)
                cov_diag = sigma_diag[mask] + (diff ** 2).mean(dim=0)
                cov_c_batch = cov_diag.mean(dim=0)
                n_c = int(mask.sum().item())
                if self.running_count[c] == 0:
                    self.running_mean[c] = mu_c_batch
                    self.running_cov[c] = cov_c_batch
                else:
                    alpha = self.momentum
                    self.running_mean[c] = alpha * self.running_mean[c] + (1 - alpha) * mu_c_batch
                    self.running_cov[c] = alpha * self.running_cov[c] + (1 - alpha) * cov_c_batch
                self.running_count[c] += n_c

    def sync_prototypes(self):
        self.prototype_mean.copy_(self.running_mean)
        self.prototype_cov.copy_(self.running_cov)
        self.prototype_count.copy_(self.running_count)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        update_prototypes: bool = True,
    ) -> torch.Tensor:
        log_sigma_diag = self.uncertainty_head(features)
        sigma_diag = torch.exp(log_sigma_diag.clamp(max=10.0))
        mu_i = features
        if update_prototypes and self.training:
            self.update_prototypes(features, labels)
        distances = []
        for c in range(self.num_classes):
            mu_c = self.prototype_mean[c]
            cov_c = self.prototype_cov[c]
            mean_dist = torch.sum((mu_i - mu_c.unsqueeze(0)) ** 2, dim=1)
            uncertainty_penalty = self.lambda_uncertainty * torch.sum(sigma_diag + cov_c.unsqueeze(0), dim=1)
            distances.append(mean_dist + uncertainty_penalty)
        distances = torch.stack(distances, dim=1)
        pos_distances = distances[torch.arange(features.size(0), device=distances.device), labels]
        log_sum_exp = torch.logsumexp(-distances / self.temperature, dim=1)
        pos_log_prob = -pos_distances / self.temperature
        return (log_sum_exp - pos_log_prob).mean()


CrossEntropyLoss = nn.CrossEntropyLoss
FocalLossBase = FocalLoss
FocalLossProCo = FocalLoss

__all__ = [
    "CrossEntropyLoss",
    "FocalLoss",
    "FocalLossBase",
    "FocalLossProCo",
    "ProbabilisticContrastiveLoss",
]
