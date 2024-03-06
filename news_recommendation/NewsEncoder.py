import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from AdditiveAttention import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "bert-base-uncased",
        multihead_attn_num_heads: int = 16,
        additive_attn_hidden_dim: int = 200,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)

        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=plm_hidden_size, num_heads=multihead_attn_num_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(plm_hidden_size, additive_attn_hidden_dim)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        output = self.plm(input_val).last_hidden_state[:, 0, :]
        return output
