import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from AdditiveAttention import AdditiveAttention


class my_PLM4NR(nn.Module):
    def __init__(
        self,
        news_encoder: nn.Module,
        interest_encoder: nn.Module,
        user_encoder: nn.Module,
        hidden_size: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        super().__init__()
        self.news_encoder: nn.Module = news_encoder
        self.interest_encoder: nn.Module = interest_encoder
        self.user_encoder: nn.Module = user_encoder

        self.news_atten = AdditiveAttention(user_encoder.hidden_size, user_encoder.additive_attn_hidden_dim)

        self.hidden_size: int = hidden_size
        self.loss_fn = loss_fn

    def forward(
        self, user_interest:torch.Tensor, candidate_news_info: torch.Tensor, history_news_info: torch.Tensor, target: torch.Tensor,
            candidate_news_topics: torch.Tensor=None, history_news_topics: torch.Tensor=None
    ) -> torch.Tensor:

        batch_size, candidate_num, seq_len = candidate_news_info.size()
        candidate_news_info = candidate_news_info.view(batch_size * candidate_num, seq_len)
        candidate_news_info_encoded = self.news_encoder(
            candidate_news_info
        )

        candidate_news_topics = candidate_news_topics.view(batch_size * candidate_num, -1)
        candidate_news_topics_encoded = self.news_encoder(
            candidate_news_topics
        )
        candidate_news_encoded = torch.stack([candidate_news_info_encoded, candidate_news_topics_encoded], dim=1)
        candidate_news_encoded = self.news_atten(candidate_news_encoded)
        news_candidate_encoded = torch.sum(candidate_news_encoded, dim=1)
        news_candidate_encoded = news_candidate_encoded.view(batch_size, candidate_num, self.hidden_size)

        u_batch_size, _, u_seq_len = user_interest.size()
        user_interest = user_interest.view(u_batch_size, u_seq_len)
        user_interest_encoded = self.interest_encoder(
            user_interest
        )
        news_histories_encoded = self.user_encoder(
            user_interest_encoded, history_news_info, history_news_topics, self.news_encoder
        )
        news_histories_encoded = news_histories_encoded.unsqueeze(
            -1
        )
        output = torch.bmm(
            news_candidate_encoded, news_histories_encoded
        )

        output = output.squeeze(-1).squeeze(-1)

        if not self.training:
            return ModelOutput(logits=output, loss=torch.Tensor([-1]), labels=target)

        loss = self.loss_fn(output, target)
        return ModelOutput(logits=output, loss=loss, labels=target)
