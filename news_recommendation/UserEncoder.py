import torch
from torch import nn
from AdditiveAttention import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        multihead_attn_num_heads: int = 16,
        additive_attn_hidden_dim: int = 200,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.additive_attn_hidden_dim = additive_attn_hidden_dim
        self.news_atten = AdditiveAttention(hidden_size, additive_attn_hidden_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=multihead_attn_num_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(hidden_size, additive_attn_hidden_dim)
        self.interest_additive_attention = AdditiveAttention(hidden_size, additive_attn_hidden_dim)

    def forward(self, user_interest_encoded: torch.Tensor, news_histories_info: torch.Tensor, news_histories_topics: torch.Tensor, news_encoder: nn.Module) -> torch.Tensor:
        batch_size, hist_size, seq_len = news_histories_info.size()
        news_histories_info = news_histories_info.view(batch_size * hist_size, seq_len)
        news_histories_info_encoded = news_encoder(
            news_histories_info
        )
        news_histories_topics = news_histories_topics.view(batch_size * hist_size, -1)
        news_histories_topics_encoded = news_encoder(
            news_histories_topics
        )
        news_histories_encoded = torch.stack([news_histories_info_encoded, news_histories_topics_encoded], dim=1)
        news_histories_encoded = self.news_atten(news_histories_encoded)
        news_histories_encoded = torch.sum(news_histories_encoded, dim=1).view(batch_size, hist_size,
                                                                               self.hidden_size)


        multihead_attn_output, _ = self.multihead_attention(
            news_histories_encoded, news_histories_encoded, news_histories_encoded
        )
        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )
        his_output = torch.sum(additive_attn_output, dim=1)
        output = self.interest_additive_attention(torch.stack([his_output, user_interest_encoded], dim=1))
        output = torch.sum(output, dim=1)
        return output
