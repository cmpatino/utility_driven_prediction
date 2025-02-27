import itertools
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import datasets
import losses


@dataclass
class Config:
    probabilities_ids: list[str]
    lambdas: list[float]
    loss_functions: list[str]
    alphas: list[float]
    n_trials: int
    random_seed: int = 42


def get_predictions(probabilities_id):
    predictions_path = Path("predictions") / probabilities_id
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file {predictions_path} does not exist.")

    all_softmax = np.load(predictions_path / "model_predictions.npy")
    all_labels = np.load(predictions_path / "labels.npy")
    with open(predictions_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    dataset = metadata["dataset"]

    return all_softmax, all_labels, dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="Run experiments with JSON configuration.")
    parser.add_argument("config", help="Path to the JSON configuration file")
    parser.add_argument(
        "--log_parquet",
        help="Whether to log results to a CSV file",
        action="store_true",
    )
    parser.add_argument(
        "--wandb",
        help="Whether to log results to Weights and Biases",
        action="store_true",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    config = Config(**config_dict)

    rng = np.random.default_rng(config.random_seed)

    param_grid = {
        "probabilities_id": config.probabilities_ids,
        "loss_functions": config.loss_functions,
        "alphas": config.alphas,
        "lambdas": config.lambdas,
        "trial_idx": range(config.n_trials),
    }

    param_combinations = list(itertools.product(*param_grid.values()))
    processed_loss_types = set()
    dfs = []
    prev_dataset = None
    prev_lambda = None
    prev_alpha = None
    prev_loss_function = None
    for probabilities_id, loss_function, alpha, lambda_, trial_idx in tqdm(param_combinations):
        # Get predicted softmax
        all_softmax, all_labels, dataset_name = get_predictions(probabilities_id)
        num_samples, num_classes = all_softmax.shape

        if dataset_name not in [
            "CIFAR100",
            "Fitzpatrick",
            "iNaturalist",
            "ImageNet",
            "FitzpatrickClean",
        ]:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        # Avoid loading the dataset again if it's the same from the previous iteration
        if prev_dataset != dataset_name:
            dataset = getattr(datasets, dataset_name)(
                embeddings=None,
                labels=all_labels,
                rng=rng,
            )
            prev_dataset = dataset_name

        # Create calibration indicator
        n_validation = (all_softmax.shape[0] - dataset.calibration_size) // 2
        n_test = all_softmax.shape[0] - dataset.calibration_size - n_validation
        # import pdb; pdb.set_trace()
        calibration_indicator_base = np.array(
            [0] * dataset.calibration_size + [1] * n_validation + [2] * n_test
        )
        calibration_indicator = rng.choice(
            calibration_indicator_base,
            len(calibration_indicator_base),
            replace=False,
        )

        loss = getattr(losses, loss_function)(dataset.calibration_size, dataset.second_level)
        if (
            loss.is_lambda_independent
            and (probabilities_id, trial_idx, loss_function, alpha) in processed_loss_types
        ):
            continue

        # Get penalties
        penalties, pairwise_distances = dataset.get_penalties(all_labels, loss.is_separable)

        if prev_lambda != lambda_ or prev_alpha != alpha or prev_loss_function != loss_function:
            scores = loss.get_scores(
                penalties=penalties,
                pairwise_distances=pairwise_distances,
                probabilities=all_softmax,
                lambda_=lambda_,
                labels=all_labels,
            )
            prev_lambda = lambda_
            prev_alpha = alpha
            prev_loss_function = loss_function
        cal_scores = scores[calibration_indicator == 0]
        val_labels = all_labels[calibration_indicator == 1]
        val_softmax = all_softmax[calibration_indicator == 1]
        val_penalties = penalties[calibration_indicator == 1]
        test_labels = all_labels[calibration_indicator == 2]
        test_softmax = all_softmax[calibration_indicator == 2]
        test_penalties = penalties[calibration_indicator == 2]

        # Calculate accuracy
        val_accuracy = accuracy_score(val_labels, np.argmax(val_softmax, axis=1))
        test_accuracy = accuracy_score(test_labels, np.argmax(test_softmax, axis=1))

        q_hat = loss.get_quantile(cal_scores, alpha)
        val_prediction_sets = loss.get_prediction_sets(
            penalties=val_penalties,
            pairwise_distances=pairwise_distances,
            probabilities=val_softmax,
            lambda_=lambda_,
        )
        test_prediction_sets = loss.get_prediction_sets(
            penalties=test_penalties,
            pairwise_distances=pairwise_distances,
            probabilities=test_softmax,
            lambda_=lambda_,
        )

        val_granular_coverage = val_prediction_sets[np.arange(n_validation), val_labels]
        test_granular_coverage = test_prediction_sets[np.arange(n_test), test_labels]

        val_empirical_coverage = val_granular_coverage.mean()
        test_empirical_coverage = test_granular_coverage.mean()

        log = {
            "val_empirical_coverage": val_empirical_coverage,
            "test_empirical_coverage": test_empirical_coverage,
            "quantile": q_hat,
            "lambda": lambda_,
            "1 - alpha": 1 - alpha,
        }

        val_evaluation = loss.eval_sets(val_prediction_sets, val_penalties)
        test_evaluation = loss.eval_sets(test_prediction_sets, test_penalties)

        # Iterate over metrics stored in the evaluation object
        for evaluation_criteria in val_evaluation.__dict__.keys():
            if getattr(val_evaluation, evaluation_criteria) is None:
                continue
            metric = getattr(val_evaluation, evaluation_criteria)

            additional_log = {
                f"val_{evaluation_criteria}_mean": metric.mean(),
                f"val_{evaluation_criteria}_std": metric.std(),
                f"val_{evaluation_criteria}_median": np.median(metric),
                f"val_{evaluation_criteria}_max": metric.max(),
                f"val_{evaluation_criteria}_min": metric.min(),
                f"val_{evaluation_criteria}_mode": stats.mode(metric).mode,
            }
            log.update(additional_log)

        for evaluation_criteria in test_evaluation.__dict__.keys():
            if getattr(test_evaluation, evaluation_criteria) is None:
                continue
            metric = getattr(test_evaluation, evaluation_criteria)

            additional_log = {
                f"test_{evaluation_criteria}_mean": metric.mean(),
                f"test_{evaluation_criteria}_std": metric.std(),
                f"test_{evaluation_criteria}_median": np.median(metric),
                f"test_{evaluation_criteria}_max": metric.max(),
                f"test_{evaluation_criteria}_min": metric.min(),
                f"test_{evaluation_criteria}_mode": stats.mode(metric).mode,
            }
            log.update(additional_log)

        processed_loss_types.add((probabilities_id, trial_idx, loss_function, alpha))

        if args.wandb:
            wandb.init(
                project="uq_decision_making",
                entity="cmpatino",
                name=f"{dataset}_{loss_function}_{alpha}_{lambda_}",
                config={
                    "dataset": dataset,
                    "probabilities_id": config.probabilities_ids,
                    "loss_function": loss_function,
                },
            )
            wandb.log(log)
            wandb.finish()

        if args.log_parquet:
            val_metrics_df = pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "trial_idx": trial_idx,
                    "size": val_evaluation.size,
                    "sum_penalties": val_evaluation.sum_penalties,
                    "max_penalties": val_evaluation.max_penalties,
                    "coverage_loss": val_evaluation.coverage_loss,
                    "lambda": lambda_,
                    "loss_function": loss_function,
                    "alpha": alpha,
                    "empirical_coverage": val_empirical_coverage,
                    "fold": "validation",
                    "probabilities_id": probabilities_id,
                    "accuracy": val_accuracy,
                    "granular_coverage": val_granular_coverage,
                }
            )
            test_metrics_df = pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "trial_idx": trial_idx,
                    "size": test_evaluation.size,
                    "sum_penalties": test_evaluation.sum_penalties,
                    "max_penalties": test_evaluation.max_penalties,
                    "coverage_loss": test_evaluation.coverage_loss,
                    "lambda": lambda_,
                    "loss_function": loss_function,
                    "alpha": alpha,
                    "empirical_coverage": test_empirical_coverage,
                    "fold": "test",
                    "probabilities_id": probabilities_id,
                    "accuracy": test_accuracy,
                    "granular_coverage": test_granular_coverage,
                }
            )
            dfs.append(val_metrics_df)
            dfs.append(test_metrics_df)

    if args.log_parquet:
        timestamp = pd.Timestamp.now().strftime("%m%d%H%M")
        # Create directory with timestamp
        Path("results").mkdir(exist_ok=True)
        (Path("results") / timestamp).mkdir(exist_ok=True)
        artifacts_dir = Path("results") / timestamp

        metrics_df = pd.concat(dfs)
        metrics_df.to_parquet(artifacts_dir / "size_loss.parquet", index=False)

        # Save config
        with open(artifacts_dir / "config.json", "w") as f:
            json.dump(config_dict, f)
        print(f"Results saved in {artifacts_dir}")
