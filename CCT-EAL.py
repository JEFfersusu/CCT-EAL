import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CCTEntropyAlignLoss(nn.Module):
    """
    CCT-EAL: Class-Conditional Temperature with Entropy Alignment Loss

    L = CE(z / T_c, y) + alpha_t * ( H_pred - H* )^2

    - Class-conditional temperature:
        T_c = softplus(tau_c), clamped to [T_min, T_max]
    - Entropy alignment target H*:
        Entropy of EMA-smoothed label frequency
    - alpha_t:
        Warmed up externally via set_progress()
    """

    def __init__(
        self,
        num_classes: int,
        alpha_max: float = 8.0,
        ema_momentum: float = 0.9,
        T_min: float = 0.75,
        T_max: float = 1.5,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.alpha_max = float(alpha_max)
        self.alpha_t = float(alpha_max)
        self.ema_momentum = float(ema_momentum)
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.eps = float(eps)

        # Initialize tau so that softplus(tau) ≈ 1.0
        tau_init = torch.log(torch.exp(torch.tensor(1.0)) - 1.0)
        self.tau_c = nn.Parameter(
            tau_init * torch.ones(self.num_classes)
        )  # [C]

        # EMA of label distribution (initialized as uniform)
        self.register_buffer(
            "label_freq_ema",
            torch.full((self.num_classes,), 1.0 / self.num_classes),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_progress(self, progress: float, warmup_ratio: float = 0.3):
        """
        Linearly warm up alpha_t from 0 to alpha_max during early training.

        Args:
            progress: training progress in [0, 1]
            warmup_ratio: fraction of training for warm-up
        """
        progress = float(max(0.0, min(1.0, progress)))
        if progress < warmup_ratio:
            self.alpha_t = self.alpha_max * (progress / max(warmup_ratio, 1e-6))
        else:
            self.alpha_t = self.alpha_max

    @torch.no_grad()
    def _update_label_freq(self, targets: torch.Tensor):
        """
        Update EMA of label frequencies using current mini-batch.
        """
        counts = torch.bincount(
            targets, minlength=self.num_classes
        ).float()
        batch_freq = counts / max(targets.numel(), 1)

        self.label_freq_ema.mul_(self.ema_momentum).add_(
            batch_freq, alpha=1.0 - self.ema_momentum
        )

    @torch.no_grad()
    def _label_entropy(self) -> torch.Tensor:
        """
        Compute entropy of EMA label distribution.
        """
        p = self.label_freq_ema.clamp_min(self.eps)
        return -(p * p.log()).sum()

    def _class_temperature(self) -> torch.Tensor:
        """
        Compute class-conditional temperatures T_c.
        """
        T = F.softplus(self.tau_c)
        return T.clamp(self.T_min, self.T_max)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        update_stats: bool = True,
    ):
        """
        Args:
            logits: [B, C] unnormalized model outputs
            targets: [B] ground-truth labels
            update_stats: whether to update EMA (True for training, False for eval)

        Returns:
            loss: scalar tensor
            stats: dictionary of monitoring values
        """
        targets = targets.long().to(logits.device)

        # Update label statistics (training only)
        if update_stats:
            self._update_label_freq(targets)

        # Target entropy H*
        H_star = self._label_entropy()

        # Class-conditional temperature scaling
        T_c = self._class_temperature()              # [C]
        T = T_c[targets].unsqueeze(1)                # [B, 1]
        logits_scaled = logits / T

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_scaled, targets)

        # Predictive entropy H_pred
        log_probs = F.log_softmax(logits_scaled, dim=1)
        probs = log_probs.exp()
        H_pred = -(probs * log_probs).sum(dim=1).mean()

        # Entropy alignment loss
        ea_loss = (H_pred - H_star).pow(2)

        # Total loss
        loss = ce_loss + self.alpha_t * ea_loss

        stats: Dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "ce": ce_loss.detach(),
            "H_pred": H_pred.detach(),
            "H_star": H_star.detach(),
            "alpha_t": torch.tensor(self.alpha_t, device=logits.device),
            "T_mean": T.mean().detach(),
        }

        return loss, stats

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def scale_logits(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Apply class-conditional temperature scaling for evaluation metrics.
        """
        targets = targets.long().to(logits.device)
        T_c = self._class_temperature()
        T = T_c[targets].unsqueeze(1)
        return logits / T
