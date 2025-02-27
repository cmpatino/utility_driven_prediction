import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


@dataclass
class TrainingDataset:
    dataset: Dataset
    num_classes: int


class FitzpatrickDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.root_dir / f"{self.data.iloc[idx]['md5hash']}.jpg"
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label_int"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_fitzpatrick(split: Literal["training", "prediction"], transforms) -> TrainingDataset:
    slurm_user = os.environ.get("SLURM_JOB_USER")
    if slurm_user:
        root_dir = f"/data/user_data/{slurm_user}"
    else:
        root_dir = "../data"
    if split == "training":
        dataset = FitzpatrickDataset(
            csv_file=f"{root_dir}/fitzpatrick/train.csv",
            root_dir=f"{root_dir}/fitzpatrick/images/",
            transform=transforms,
        )
    elif split == "prediction":
        dataset = FitzpatrickDataset(
            csv_file=f"{root_dir}/fitzpatrick/test.csv",
            root_dir=f"{root_dir}/fitzpatrick/images/",
            transform=transforms,
        )
    else:
        raise ValueError(f"Incorrect split: {split}")
    return TrainingDataset(dataset=dataset, num_classes=114)


def get_fitzpatrick_clean(split: Literal["training", "prediction"], transforms) -> TrainingDataset:
    slurm_user = os.environ.get("SLURM_JOB_USER")
    if slurm_user:
        root_dir = f"/data/user_data/{slurm_user}"
    else:
        root_dir = "../data"
    if split == "training":
        dataset = FitzpatrickDataset(
            csv_file=f"{root_dir}/fitzpatrick_clean/train.csv",
            root_dir=f"{root_dir}/fitzpatrick_clean/images/",
            transform=transforms,
        )
    elif split == "prediction":
        dataset = FitzpatrickDataset(
            csv_file=f"{root_dir}/fitzpatrick_clean/test.csv",
            root_dir=f"{root_dir}/fitzpatrick_clean/images/",
            transform=transforms,
        )
    else:
        raise ValueError(f"Incorrect split: {split}")
    return TrainingDataset(dataset=dataset, num_classes=dataset.num_classes)


def get_cifar100(split: Literal["training", "prediction"], transforms) -> TrainingDataset:
    slurm_user = os.environ.get("SLURM_JOB_USER")
    if slurm_user:
        root_dir = f"/data/user_data/{slurm_user}"
    else:
        root_dir = "../data"
    if split == "training":
        dataset = datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transforms)
    elif split == "prediction":
        dataset = datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transforms)
    else:
        raise ValueError(f"Incorrect split: {split}")
    return TrainingDataset(dataset=dataset, num_classes=len(dataset.classes))


def get_inaturalist_class(split: Literal["training", "prediction"], transforms) -> TrainingDataset:
    slurm_user = os.environ.get("SLURM_JOB_USER")
    if slurm_user:
        root_dir = Path(f"/data/user_data/{slurm_user}")
    else:
        root_dir = Path("../data")
    if split == "training":
        dataset = datasets.INaturalist(
            root=root_dir / "inaturalist",
            version="2021_train_mini",
            transform=transforms,
            target_type="class",
        )
    elif split == "prediction":
        dataset = datasets.INaturalist(
            root=root_dir / "inaturalist",
            version="2021_valid",
            transform=transforms,
            target_type="class",
        )
    else:
        raise ValueError(f"Incorrect split: {split}")
    return TrainingDataset(dataset=dataset, num_classes=51)


def get_imagenet(split: Literal["training", "prediction"], transforms) -> TrainingDataset:
    slurm_user = os.environ.get("SLURM_JOB_USER")
    if slurm_user:
        root_dir = Path(f"/data/user_data/{slurm_user}")
    else:
        root_dir = Path("../data")
    if split == "training":
        dataset = datasets.ImageNet(
            root=root_dir / "imagenet",
            split="train",
            transform=transforms,
        )
    elif split == "prediction":
        dataset = datasets.ImageNet(
            root=root_dir / "imagenet",
            split="val",
            transform=transforms,
        )
    else:
        raise ValueError(f"Incorrect split: {split}")
    return TrainingDataset(dataset=dataset, num_classes=1000)
