import itertools
import json
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm


@dataclass
class Config:
    probabilities_ids: list[str]
    lambdas: list[float]
    loss_functions: list[str]
    alphas: list[float]
    n_trials: int
    random_seed: int = 42


@dataclass
class Artifacts:
    all_softmax: np.ndarray
    all_labels: np.ndarray
    dataset_name: str
    hierarchy_artifacts: dict


def get_artifacts(probabilities_id):
    predictions_path = Path("utility_input_artifacts") / probabilities_id
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file {predictions_path} does not exist.")

    all_softmax = np.load(predictions_path / "model_predictions.npy")
    all_labels = np.load(predictions_path / "labels.npy")
    with open(predictions_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    dataset_name = metadata["dataset"]
    if dataset_name not in [
        "iNaturalist",
        "Fitzpatrick",
        "FitzpatrickClean",
        "ImageNet",
        "CIFAR100",
    ]:
        raise ValueError(f"Dataset {dataset_name} read from metadata is not supported.")

    hierarchy_artifacts = {}
    if dataset_name == "iNaturalist":
        with open('hierarchy_artifacts/inaturalist/species_list.json', 'r') as file:
            all_categories = json.load(file)

        with open('hierarchy_artifacts/inaturalist/mapping_dict.json', 'r') as file:
            categories_index = json.load(file)

        hierarchy_artifacts["all_categories"] = all_categories
        hierarchy_artifacts["categories_index"] = categories_index
    elif dataset_name == "Fitzpatrick":
        hierarchy_artifacts["label_hierarchy"] = np.load(
            "hierarchy_artifacts/fitzpatrick/fitzpatrick_hierarchy.npy"
        )
    elif dataset_name == "FitzpatrickClean":
        hierarchy_artifacts["label_hierarchy"] = np.load(
            "hierarchy_artifacts/fitzpatrick/fitzpatrick_clean_hierarchy.npy"
        )
    elif dataset_name in ["ImageNet", "CIFAR100"]:
        hierarchy_artifacts["embeddings"] = np.load(predictions_path / "embeddings.npy")

    artifacts = Artifacts(all_softmax, all_labels, dataset_name, hierarchy_artifacts)
    return artifacts


def dict_to_list(leaf_nodes):
    return [leaf_nodes[key] for key in leaf_nodes.keys()]


def profile_and_animate(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        running = True

        def blink_message():
            for c in itertools.cycle(["|", "/", "-", "\\"]):
                if not running:
                    break
                sys.stdout.write(
                    f"\rComputing non-conformity scores. This process may take a few minutes, please wait {c}"
                )
                sys.stdout.flush()
                time.sleep(0.2)
            sys.stdout.write("\rComputation completed!                                \n")

        t = threading.Thread(target=blink_message)
        t.start()

        result = func(*args, **kwargs)

        running = False
        t.join()

        end = time.time()
        print(f"Time elapsed: {end - start:.2f} seconds")
        return result

    return wrapper


# @profile_and_animate
def compute_parallel_process(function, iterable, paralel=True):
    # TODO Usefull for debugging
    if not paralel:
        predictions_set = []
        for i in tqdm(iterable):
            output = function(i)
            predictions_set.append(output)
    else:
        with Pool() as pool:
            predictions_set = pool.map(function, iterable)

    return predictions_set


class TreeNode:
    def __init__(self, value=0, children=None):
        self.value = value
        self.children = children if children is not None else []
