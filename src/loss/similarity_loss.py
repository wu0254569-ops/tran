"""The custom loss: pull the model's action-position representation toward the
embedding of the action it should have taken.

See 系统提示词/核心.md and 系统提示词.json ("核心机制").

    L = 1 - cos_sim(h_wrong, emb(right))

- h_wrong: the hidden state at the decision position (the context in which the
  wrong action was chosen).
- emb(right): the token embedding of the correct action word (frozen).

The embedding / lm_head matrix ("语义特征层") is frozen so we never collapse
different actions into similar vectors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def similarity_loss(
    predicted_repr: torch.Tensor,
    target_embedding: torch.Tensor,
    metric: str = "cosine",
    direction: str = "pull",
) -> torch.Tensor:
    """Return a scalar loss in float32.

    `predicted_repr` and `target_embedding` are both [hidden] vectors.
    direction="pull" minimizes distance; "push" maximizes it.
    """
    if predicted_repr.dim() == 1:
        predicted_repr = predicted_repr.unsqueeze(0)
    if target_embedding.dim() == 1:
        target_embedding = target_embedding.unsqueeze(0)

    if metric == "cosine":
        similarity = F.cosine_similarity(
            predicted_repr.float(), target_embedding.float(), dim=-1
        )
    elif metric == "dot":
        similarity = (predicted_repr.float() * target_embedding.float()).sum(dim=-1)
    elif metric == "mse":
        similarity = -(predicted_repr.float() - target_embedding.float()).pow(2).mean(dim=-1)
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    similarity = similarity.mean()
    return 1.0 - similarity if direction == "pull" else similarity
