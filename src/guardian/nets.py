"""
Actor-critic network for the optional AlphaZero-style distillation, acts as a computationally
cheaper proposer.

A small shared-trunk MLP with two heads:
- Policy head (cross-entropy) which imitates the shielded expert's action,
- Value head (MSE) which predicts the search's safety value.
"""

import os

import torch
import torch.nn as nn

from . import config as C
from . import dynamics as D

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POLICY = os.path.join(_REPO, "data", "guardian_policy.pt")


class ActorCritic(nn.Module):
    """MLP (Multilayer Perceptron) with a policy head (2 logits) and a value head (scalar in [-1, 1])."""

    def __init__(self, in_dim: int = 5, hidden: int = C.HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, 2)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(policy_logits, value)` for a batch of features."""
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)

    @torch.no_grad()
    def act(self, y: int, vel: int, pipes: list) -> int:
        """Greedy reactive action from the policy head (no search, no shield)."""
        x = torch.from_numpy(D.features(y, vel, pipes)).unsqueeze(0)
        logits, _ = self.forward(x)
        return int(logits.argmax(-1).item())


def load_policy(path: str | None = None) -> ActorCritic:
    """
    Load a distilled policy net for deployment in eval mode, and deploy
    it as a proposer for the shield.
    """
    net = ActorCritic()
    net.load_state_dict(torch.load(path or DEFAULT_POLICY, map_location="cpu"))
    net.eval()
    return net


def make_leaf_value(net: ActorCritic):
    """
    Make a leaf-scoring function for the depth-limited search.
    Score leaves with the network's value head, vs just current clearance full search.
    """
    @torch.no_grad()
    def leaf_value(y: int, vel: int, pipes: list) -> float:
        x = torch.from_numpy(D.features(y, vel, pipes)).unsqueeze(0)
        _, v = net.forward(x)
        return float(v.item())

    return leaf_value
