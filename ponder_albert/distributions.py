import torch
from torch import nn
from typing import Union


class GeneralizedGeometricDist(nn.Module):
    def __init__(self,
                 stop_probabilities: Union[torch.Tensor, float],
                 max_steps: Union[None, int] = None,
                 batch_size: Union[None, int] = None,
                 device: str = 'cpu'):
        """Generalized Geometric Distribution
        Args:
            stop_probabilities (Union[torch.Tensor, float]): Either a tensor
                of dimensions (# texts, # layers) or a float in [0, 1].
                If a float is passed,  `max_steps` and `batch_size` must
                be provided.
            max_steps (Union[None, int]], optional): Max number of steps of the
                support of the geometric distribution. Must be provided if
                `stop_probabilities` is not a tensor. Defaults to None.
            batch_size (Union[None, int], optional): Batch size.
                Must be provided if `stop_probabilities` is not a tensor.
                Defaults to None.
            device (str, optional): Defaults to 'cpu'.
        """

        super().__init__()
        self.device = device

        # If a float is passed, broadcasts `stop_probabilities`
        # to a tensor of sizes (batch_size, max_steps).
        if isinstance(stop_probabilities, float):
            assert batch_size is not None
            assert max_steps is not None

            stop_probabilities = torch.tensor(
                [stop_probabilities] * max_steps,
                dtype=torch.float32,
                device=device
            )

            stop_probabilities = stop_probabilities.expand(batch_size, -1)

        self.stop_probabilities = stop_probabilities

    def pmf(self, normalize=True):
        """Probability Mass Function

        Args:
            normalize (bool, optional): Normalizes distribution, effectively
                dividing the output by output.sum(axis=0). This is the same as
                computing the conditional probability P(X = x | X <= max_steps)
                of a geometric distribution X with infinite support.
                Defaults to True.

        Returns:
            [torch.Tensor]: Probability mass function evaluated at
                x = 1, ..., max_steps.
        """
        keep_probs = 1 - self.stop_probabilities
        keep_probs_acc = keep_probs.cumprod(axis=1)

        # Shifts `keep_probs_acc` to the right and adds a column w/ ones
        keep_probs_acc = torch.cat([
            torch.ones(keep_probs.shape[0], 1), keep_probs_acc[:, :-1]
        ], axis=1)

        pmf = keep_probs_acc * self.stop_probabilities

        if normalize:
            return pmf / pmf.sum(axis=1)[..., None]

        return pmf

    def kl_div(self, target: 'GeneralizedGeometricDist'):  # Python 3.6 compatibility
        """Computes the KL divergence KL(p || q) between
            two `GeneralizedGeometricDist`.

        Args:
            target (GeneralizedGeometricDist): The second probability
                distribution (q) in KL(p || q).

        Returns:
            torch.Tensor: KL(self || target)
        """
        p, q = self.pmf(), target.pmf()
        return torch.nn.functional.kl_div(p.log(), q, reduction='batchmean')
