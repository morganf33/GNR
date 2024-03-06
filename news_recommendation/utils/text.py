from typing import Callable
from transformers import PreTrainedTokenizer
import torch
from typing import List

def create_transform_fn_from_pretrained_tokenizer(
    tokenizer: PreTrainedTokenizer, train_mode: bool
) -> Callable[[List[str], str], torch.Tensor]:
    if train_mode:
        max_length = [100, 20, 50]
    else:
        max_length = [256, 20, 50]
    def transform(texts: List[str], mode: str="title") -> torch.Tensor:
        if mode=="title+abstract":
            output = tokenizer(texts, return_tensors="pt", max_length=max_length[0], padding="max_length", truncation=True)["input_ids"]
        elif mode=="topics":
            output = tokenizer(texts, return_tensors="pt", max_length=max_length[1], padding="max_length", truncation=True)["input_ids"]
        elif mode == "interest":
            output = tokenizer(texts, return_tensors="pt", max_length=max_length[2], padding="max_length", truncation=True)["input_ids"]
        return output
    return transform
