import torch
from torch import nn


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(
                input_dim, hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim, 1, bias=False
            ),
            nn.Softmax(dim=-2),
        )
        self.attention.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attention_weight = self.attention(input)
        return input * attention_weight
