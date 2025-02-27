from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering


@dataclass
class Hierarchy:
    """Defines the structure of a hierarchy with root -> 1st level -> 2nd level -> classes.

    We impose this hierarchy in the code so that we have a comparable hierarchy to the Fitzpatrick
    hierarchy. However, our approach should generalize well to arbitrary hierarchies.
    """

    second_level: dict[str, list[int]]
    first_level: list[int]


class ClusteringHierarchy:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = labels
        self.n_second_level = 9
        self.n_first_level = 3

    def get_hierarchy(self) -> Hierarchy:
        second_level_embeddings = self._get_mean_embeddings(self.embeddings, self.labels)
        second_level_clusters = self._get_clusters(second_level_embeddings, self.n_second_level)
        second_level_hierarchy = [[] for _ in range(self.n_second_level)]
        third_to_second = {}  # Class ->  Second level cluster label
        for i, cluster in enumerate(second_level_clusters):
            second_level_hierarchy[cluster].append(i)
            third_to_second[i] = cluster

        # Assign labels from the second level cluster to each row
        first_level_labels = np.zeros_like(self.labels)
        for i, label in enumerate(self.labels):
            first_level_labels[i] = third_to_second[label]

        first_level_embeddings = self._get_mean_embeddings(self.embeddings, first_level_labels)
        first_level_clusters = self._get_clusters(first_level_embeddings, self.n_first_level)

        consecutive_first_level = np.argsort(first_level_clusters)
        first_level = first_level_clusters[consecutive_first_level]
        consecutive_hierarchy = [None for _ in range(self.n_second_level)]

        for new_idx, orig_idx in enumerate(consecutive_first_level):
            consecutive_hierarchy[new_idx] = second_level_hierarchy[orig_idx]

        hierarchy = {}
        for i in range(self.n_second_level):
            hierarchy[f"{i}"] = consecutive_hierarchy[i]

        return Hierarchy(second_level=hierarchy, first_level=first_level)

    def _get_clusters(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
        return clustering.fit_predict(embeddings)

    def _get_mean_embeddings(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        class_embeddings = []
        for label_idx in sorted(np.unique(labels)):
            class_embeddings.append(embeddings[labels == label_idx].mean(axis=0))
        return np.array(class_embeddings)


class FitzpatrickHierarchy:
    def __init__(self, label_hierarchy: np.ndarray, labels: np.ndarray):
        self.label_hierarchy = label_hierarchy
        self.labels = labels
        self.n_second_level = 9
        self.n_first_level = 3

    def get_hierarchy(self) -> Hierarchy:
        third_to_second = {}  # Second level cluster label -> First level cluster label
        for row_idx in range(len(self.label_hierarchy)):
            if self.label_hierarchy[row_idx, 2] in third_to_second:
                assert (
                    third_to_second[self.label_hierarchy[row_idx, 2]]
                    == self.label_hierarchy[row_idx, 1]
                )
                continue
            else:
                third_to_second[self.label_hierarchy[row_idx, 2]] = self.label_hierarchy[row_idx, 1]
        second_level_clusters = np.array(
            [third_to_second[second] for second in sorted(third_to_second.keys())]
        )

        second_level_hierarchy = [[] for _ in range(self.n_second_level)]
        for i, cluster in enumerate(second_level_clusters):
            second_level_hierarchy[cluster].append(i)
            third_to_second[i] = cluster

        # Remove indexes with no elements
        second_level_hierarchy = [cluster for cluster in second_level_hierarchy if cluster]

        second_to_first = {}  # Second level cluster label -> First level cluster label
        for row_idx in range(len(self.label_hierarchy)):
            if self.label_hierarchy[row_idx, 1] in second_to_first:
                assert (
                    second_to_first[self.label_hierarchy[row_idx, 1]]
                    == self.label_hierarchy[row_idx, 0]
                )
                continue
            else:
                second_to_first[self.label_hierarchy[row_idx, 1]] = self.label_hierarchy[row_idx, 0]

        first_level_clusters = np.array(
            [second_to_first[second] for second in second_to_first.keys()]
        )

        consecutive_first_level = np.argsort(first_level_clusters)
        first_level = first_level_clusters[consecutive_first_level]
        consecutive_hierarchy = [None for _ in range(len(first_level_clusters))]
        for new_idx, orig_idx in enumerate(consecutive_first_level):
            consecutive_hierarchy[new_idx] = second_level_hierarchy[orig_idx]

        hierarchy = {}
        for i in range(len(first_level_clusters)):
            hierarchy[f"{i}"] = consecutive_hierarchy[i]

        return Hierarchy(second_level=hierarchy, first_level=first_level)
