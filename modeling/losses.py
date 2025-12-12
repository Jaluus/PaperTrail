import torch
import torch.nn.functional as F


def BPR_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Computes the Bayesian Personalized Ranking (BPR) loss.

    Args:
        pos_scores (Tensor): Predicted scores for positive samples.
        neg_scores (Tensor): Predicted scores for negative samples.

    Returns:
        Tensor: Computed BPR loss.
    """
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    return loss
