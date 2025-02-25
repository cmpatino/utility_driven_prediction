import datetime
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import wandb
from dotenv import load_dotenv
from model_config import Config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets

DATASET_NAME_MAP = {
    "cifar100": "CIFAR100",
    "inaturalist_class": "iNaturalist",
    "imagenet": "ImageNet",
    "fitzpatrick": "Fitzpatrick",
    "fitzpatrick_clean": "FitzpatrickClean",
}

load_dotenv()


class VisionModel(nn.Module):
    def __init__(self, backbone, weights, num_classes=100):
        super(VisionModel, self).__init__()
        if backbone == "ResNet50":
            self.backbone = models.resnet50(weights=weights)
        elif backbone == "EfficientNet_B0":
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"The backbone {backbone} is not supported")
        self.output_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    parser = ArgumentParser(description="Run experiments with JSON configuration.")
    parser.add_argument("config", help="Path to the JSON configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    config = Config(**config_dict)

    if config.vision_backbone == "ResNet50":
        pretrained_weights = models.ResNet50_Weights.DEFAULT
    elif config.vision_backbone == "EfficientNet_B0":
        pretrained_weights = models.EfficientNet_B0_Weights.DEFAULT
    else:
        raise ValueError(f"The backbone {config.vision_backbone} is not supported")

    # Get dataset to finetune backbone weights
    finetuning_dataset = getattr(datasets, f"get_{config.dataset}")(
        "training", pretrained_weights.transforms()
    )
    n_training = int(len(finetuning_dataset.dataset) * (1 - config.validation_pct))
    n_validation = int(len(finetuning_dataset.dataset) - n_training)
    random_generator = torch.Generator().manual_seed(42)
    training_dataset, validation_dataset = torch.utils.data.random_split(
        finetuning_dataset.dataset, [n_training, n_validation], generator=random_generator
    )
    num_classes = finetuning_dataset.num_classes

    # Get dataset to generate downstream set predictions
    prediction_dataset = getattr(datasets, f"get_{config.dataset}")(
        "prediction", pretrained_weights.transforms()
    ).dataset

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config=config_dict,
    )

    model = VisionModel(config.vision_backbone, pretrained_weights, num_classes=num_classes)
    wandb.watch(model)

    train_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    prediction_loader = DataLoader(
        prediction_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    softmax = nn.Softmax(dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create predictions/timestamp directory if it does not exist
    c_time = datetime.datetime.now()
    timestamp = str(c_time.strftime("%b%d-%H%M%S"))
    artifacts_dir = Path("../predictions/") / timestamp
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_path = artifacts_dir / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    num_epochs = config.epochs
    total_train_samples = len(training_dataset)
    total_validation_samples = len(validation_dataset)
    train_loss_history = []
    validation_loss_history = []
    train_accuracy_history = []
    validation_accuracy_history = []
    best_validation_loss = float("inf")
    torch.save(model.state_dict(), checkpoints_path / "model.pth")

    for epoch in tqdm(range(num_epochs), desc="Fine-tuning model"):
        train_running_loss = 0.0
        validation_running_loss = 0.0
        train_correct = 0
        validation_correct = 0
        model.train(True)
        for inputs, labels in tqdm(train_loader, leave=False, desc="Epoch training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            logits, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        with torch.inference_mode():
            for images, labels in tqdm(
                validation_loader, desc="Generating validation predictions", leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                logits, predicted = torch.max(outputs, 1)
                probabilities = softmax(outputs)
                validation_correct += (predicted == labels).sum().item()
                val_loss = criterion(outputs, labels)
                validation_running_loss += val_loss.item()

        # Log epoch checkpoint
        torch.save(model.state_dict(), checkpoints_path / f"model_epoch_{epoch}.pth")

        # Check if the model has improved and save the best model
        if validation_running_loss <= best_validation_loss:
            torch.save(model.state_dict(), checkpoints_path / "model.pth")
        else:
            break

        train_accuracy_history.append(100 * train_correct / total_train_samples)
        validation_accuracy_history.append(100 * validation_correct / total_validation_samples)
        train_loss_history.append(train_running_loss)
        validation_loss_history.append(validation_running_loss)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_running_loss,
                "validation_loss": validation_running_loss,
                "train_accuracy": train_correct / total_train_samples,
                "validation_accuracy": validation_correct / total_validation_samples,
            }
        )

    training_history = pd.DataFrame(
        {
            "epoch": range(1, num_epochs + 1),
            "train_loss": train_loss_history,
            "train_accuracy": train_accuracy_history,
            "validation_loss": validation_loss_history,
            "validation_accuracy": validation_accuracy_history,
        }
    )

    # Load the best model
    model.load_state_dict(torch.load(checkpoints_path / "model.pth"))

    # The fine-tuned model can now be used for downstream tasks.
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    all_embeddings = []

    with torch.inference_mode():  # Disable gradient calculation during inference
        for images, labels in tqdm(prediction_loader, desc="Generating predictions"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            embeddings = model.backbone(images)
            logits, predicted = torch.max(outputs, 1)
            probabilities = softmax(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all batch predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    test_accuracy = 100 * correct / total
    wandb.log({"test_accuracy": test_accuracy})
    tqdm.write(f"Test accuracy: {test_accuracy:.2f}%")

    # Save artifacts
    training_history.to_csv(artifacts_dir / "training_history.csv", index=False)
    np.save(artifacts_dir / "model_predictions.npy", all_predictions)
    np.save(artifacts_dir / "labels.npy", all_labels)
    np.save(artifacts_dir / "embeddings.npy", all_embeddings)
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    metadata = {}
    metadata["dataset"] = DATASET_NAME_MAP[config.dataset]
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Predictions saved to {artifacts_dir}.")

    wandb.finish()
