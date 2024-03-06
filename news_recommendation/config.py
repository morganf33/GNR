from dataclasses import dataclass, field

@dataclass
class PLM4NR_title_abstractConfig:
    pretrained: str = "distilbert"
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
