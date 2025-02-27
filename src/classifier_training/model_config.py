from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    vision_backbone: str
    epochs: int
    dataset: Literal[
        "cifar100", "fitzpatrick", "fitzpatrick_clean", "inaturalist_class", "imagenet"
    ]
    validation_pct: float
    learning_rate: float
    epochs: int
    batch_size: int
    compute_hierarchy: bool
