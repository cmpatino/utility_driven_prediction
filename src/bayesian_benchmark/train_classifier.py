import datetime
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import wandb
from PIL import Image
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


class BayesianLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super(BayesianLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for the weight and bias distributions
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.zeros(out_features))

        # Prior distributions for the weights and biases
        self.prior = Normal(prior_mu, prior_sigma)

    def forward(self, x):
        # Sample weights and biases from the learned posterior
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)

        return x @ weight.T + bias

    def kl_divergence(self):
        # Prior parameters
        prior_mu = self.prior.mean
        prior_sigma = self.prior.stddev

        # Posterior parameters
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # KL divergence for weights
        kl_weight = 0.5 * torch.sum(
            2 * (torch.log(prior_sigma) - self.weight_log_sigma)
            + (weight_sigma**2 + (self.weight_mu - prior_mu) ** 2) / prior_sigma**2
            - 1
        )

        # KL divergence for biases
        kl_bias = 0.5 * torch.sum(
            2 * (torch.log(prior_sigma) - self.bias_log_sigma)
            + (bias_sigma**2 + (self.bias_mu - prior_mu) ** 2) / prior_sigma**2
            - 1
        )

        return kl_weight + kl_bias


class BayesianNN(nn.Module):
    def __init__(self, backbone, weights, num_classes=100):
        super(BayesianNN, self).__init__()
        if backbone == "ResNet50":
            self.backbone = models.resnet50(weights=weights)
        elif backbone == "EfficientNet_B0":
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"The backbone {backbone} is not supported")
        self.relu = nn.ReLU()
        self.bayesian_layer = BayesianLinearLayer(1000, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.bayesian_layer(x)
        return self.softmax(x)


def approximate_conditional_distribution(model, x, num_samples=100):
    """
    Approximate p(y|x) using Monte Carlo sampling from the learned posterior.

    Args:
        model: BayesianNN with a Bayesian last layer.
        x: Input tensor (shape: batch_size x input_dim).
        num_samples: Number of posterior samples.

    Returns:
        predictions: Tensor of shape (num_samples x batch_size x output_dim),
                     representing the distribution of predictions.
    """
    predictions = []
    for _ in range(num_samples):
        # Sample weights and biases from the posterior
        output = model(x)
        predictions.append(output.detach())  # Collect predictions for each sample

    predictions = torch.stack(predictions, dim=0)  # Shape: (num_samples, batch_size, output_dim)
    return predictions


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

    prediction_dataset = FitzpatrickDataset(
        csv_file="../data/test.csv",
        root_dir="../data/finalfitz17k/",
        transform=pretrained_weights.transforms(),
    )
    finetune_dataset = FitzpatrickDataset(
        csv_file="../data/train.csv",
        root_dir="../data/finalfitz17k/",
        transform=pretrained_weights.transforms(),
    )

    wandb.init(
        project="training-FP-model",
        entity="cmpatino",
        config=config_dict,
    )

    model = BayesianNN(config.vision_backbone, pretrained_weights, num_classes=114)
    kl_weight = 0.1
    wandb.watch(model)

    train_loader = DataLoader(
        finetune_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    prediction_loader = DataLoader(
        prediction_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
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
    total_train_samples = len(finetune_dataset)
    total_test_samples = len(prediction_dataset)
    train_loss_history = []
    test_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    torch.save(model.state_dict(), checkpoints_path / "model_checkpoint_epoch_00.pth")

    step_c = 0
    for epoch in tqdm(range(num_epochs), desc="Fine-tuning model"):
        train_running_loss = 0.0
        test_running_loss = 0.0
        train_correct = 0
        test_correct = 0
        model.train(True)
        for inputs, labels in tqdm(train_loader, leave=False, desc="Epoch training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Compute KL divergence from Bayesian layer
            kl_div = model.bayesian_layer.kl_divergence()
            # Total loss (MSE + weighted KL divergence)
            loss += kl_weight * kl_div

            logits, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            step_c = +1

        with torch.inference_mode():  # Disable gradient calculation during inference
            for images, labels in tqdm(
                prediction_loader, desc="Generating predictions", leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                logits, predicted = torch.max(outputs, 1)
                probabilities = softmax(outputs)
                test_correct += (predicted == labels).sum().item()
                test_loss = criterion(outputs, labels)
                test_running_loss += test_loss.item()

        wandb.log(
            {"epoch": epoch, "train_loss": train_running_loss, "test_loss": test_running_loss}
        )

        train_accuracy_history.append(100 * train_correct / total_train_samples)
        test_accuracy_history.append(100 * test_correct / total_test_samples)
        train_loss_history.append(train_running_loss)
        test_loss_history.append(test_running_loss)

    training_history = pd.DataFrame(
        {
            "epoch": range(1, num_epochs + 1),
            "train_loss": train_loss_history,
            "train_accuracy": train_accuracy_history,
            "test_loss": test_loss_history,
            "test_accuracy": test_accuracy_history,
        }
    )

    # # The fine-tuned model can now be used for downstream tasks.
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    num_samples = 100
    with torch.inference_mode():  # Disable gradient calculation during inference
        for images, labels in tqdm(prediction_loader, desc="Generating predictions"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits, predicted = torch.max(outputs, 1)
            predictions = approximate_conditional_distribution(model, images, num_samples)
            probabilities = predictions.mean(dim=0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batch predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    test_accuracy = 100 * correct / total
    wandb.log({"test_accuracy": test_accuracy})
    tqdm.write(f"Test accuracy: {test_accuracy:.2f}%")

    # Save artifacts
    training_history.to_csv(artifacts_dir / "training_history.csv", index=False)
    np.save(artifacts_dir / "model_predictions.npy", all_predictions)
    np.save(artifacts_dir / "labels.npy", all_labels)
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    metadata = {}
    if config.dataset == "cifar100":
        metadata["dataset"] = "cifar100"
    elif config.dataset == "inaturalist_class":
        metadata["dataset"] = "iNaturalist"
    elif config.dataset == "imagenet":
        metadata["dataset"] = "ImageNet"
    elif config.dataset == "Fitzpatrick":
        metadata["dataset"] = "Fitzpatrick"
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Predictions saved to {artifacts_dir}.")

    wandb.finish()
