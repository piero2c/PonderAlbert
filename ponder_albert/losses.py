from typing import Dict
import torch
from torch import nn


class PonderClassificationLoss(nn.Module):
    def __init__(self, kl_penalty_factor: float = 5e-3):
        """PonderClassificationLoss

        PonderNet loss function:
        L = weighted cross entropy loss + KL(target_distr || model_distr) * kl_penalty_factor

        Args:
            kl_penalty_factor (float): Coefficient for KL term
        """
        super().__init__()
        self.kl_penalty_factor = kl_penalty_factor

    def forward(self, result_dict: Dict, labels: torch.Tensor):
        logits = result_dict['logits']
        model_halt_dist, target_halt_dist = result_dict['model_halt_dist'], result_dict['target_halt_dist']
        nb_layers, _, nb_labels = logits.shape

        # Stores cross entropy terms [-log(p_i), ...]
        loss_components = nn.functional.cross_entropy(
            logits.reshape(-1, nb_labels), labels.repeat(nb_layers),
            reduction='none'
        )

        # Weights each cross entropy term using `halt_dist` pmf
        weighted_loss = loss_components * model_halt_dist.pmf().T.reshape(-1)
        weighted_loss = weighted_loss.mean()

        # Computes loss terms
        kl_loss = self.kl_penalty_factor * target_halt_dist.kl_div(model_halt_dist)

        return {
            'kl_loss': kl_loss,
            'weighted_ce_loss': weighted_loss,
            'total_loss': weighted_loss + kl_loss
        }
