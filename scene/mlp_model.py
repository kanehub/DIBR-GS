from torch import nn
import torch
from torch.nn import functional as F

class GateMLP(nn.Module):
    """
    Predict per-channel gates g in (0, 1) with stable initialization.
    Input:  r_star  [N, Dr]
    Output: g       [N, Da]
    """
    def __init__(self, Dr: int, Da: int, hidden: int = 128, init_bias: float = -2.0):
        super().__init__()
        self.norm = nn.LayerNorm(Dr)
        self.fc1 = nn.Linear(Dr, hidden)
        self.fc2 = nn.Linear(hidden, Da)

        # Initialization: start with small gates
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_bias)  # sigmoid(-3)≈0.047

    def forward(self, r_star: torch.Tensor) -> torch.Tensor:
        x = self.norm(r_star)
        x = F.silu(self.fc1(x))
        g = torch.sigmoid(self.fc2(x))
        return g