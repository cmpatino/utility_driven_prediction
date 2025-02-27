from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

import hierarchies
import utils


class Dataset(ABC):
    def __init__(
        self,
        labels: np.ndarray,
        rng: np.random.Generator,
        calibration_size: int,
    ):
        self.num_classes = np.unique(labels).size
        self.num_samples = labels.size
        self.rng = rng
        self.calibration_size = calibration_size

    def get_penalties(self, labels: np.ndarray, separable: bool):
        if separable:
            return self._get_separable_penalty()
        else:
            return self._get_non_separable_penalty(labels)

    def _floyd_warshall(self, graph) -> tuple[np.ndarray, list]:
        INF = float("inf")
        num_nodes = len(graph)
        dist = [[INF] * num_nodes for _ in range(num_nodes)]

        leaf_indexes = []
        # Initialize distances based on the graph edges
        for i, node_name_i in enumerate(graph.keys()):
            if "Leaf" in node_name_i:
                label = self._get_hierarchy_label(node_name_i)
                leaf_indexes.append((i, label))
            for j, node_name_j in enumerate(graph.keys()):
                if i == j:
                    dist[i][j] = 0
                elif node_name_j in graph[node_name_i]:
                    dist[i][j] = 1 / 8
                    dist[j][i] = 1 / 8

        # Apply Floyd-Warshall algorithm
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        return np.array(dist), leaf_indexes

    def _leaf_nodes_matrix(self, leaf_indexes, distance_matrix):
        leaf_indexes_, leaf_labels = zip(*leaf_indexes)
        # TODO remove this two lines
        leaf_indexes_ = np.array(leaf_indexes_)
        leaf_labels = np.array(leaf_labels)

        distances = np.zeros((len(leaf_labels), len(leaf_labels)))
        for orig_index_i, label_i in leaf_indexes:
            for orig_index_j, label_j in leaf_indexes:
                distances[label_i, label_j] = distance_matrix[orig_index_i, orig_index_j]

        return distances

    @abstractmethod
    def _get_non_separable_penalty(self, labels: np.ndarray):
        pass

    @abstractmethod
    def _get_separable_penalty(self):
        pass

    @abstractmethod
    def _get_hierarchy_label(self, node_name: str) -> int:
        pass


class CIFAR100(Dataset):
    def __init__(
        self,
        hierarchy_artifacts: dict,
        labels: np.ndarray,
        rng: np.random.Generator,
        calibration_size: int = 25_000,
    ):
        super().__init__(labels, rng, calibration_size)
        self.super_classes = [
            "aquatic mammals",
            "fish",
            "flowers",
            "food containers",
            "fruit and vegetables",
            "household electrical devices",
            "household furniture",
            "insects",
            "large carnivores",
            "large man-made outdoor things",
            "large natural outdoor scenes",
            "large omnivores and herbivores",
            "medium-sized mammals",
            "non-insect invertebrates",
            "people",
            "reptiles",
            "small mammals",
            "trees",
            "vehicles 1",
            "vehicles 2",
        ]
        self.super_classes_labels = {
            i: self.super_classes[i] for i in range(len(self.super_classes))
        }
        embeddings = hierarchy_artifacts["embeddings"]
        hierarchy_generator = hierarchies.ClusteringHierarchy(embeddings=embeddings, labels=labels)
        hierarchy = hierarchy_generator.get_hierarchy()
        self.first_level = hierarchy.first_level
        self.second_level = hierarchy.second_level

    def _get_separable_penalty(self) -> np.ndarray:
        bins = 4
        discrete_grid = np.array(list(range(1, bins)))

        labels_array = np.tile(np.arange(self.num_classes), (self.num_samples, 1))
        super_class_penalty = self.rng.choice(discrete_grid, size=len(self.super_classes))
        penalties_array = np.vectorize(lambda label: super_class_penalty[label])
        super_class_array = labels_array // 5
        return penalties_array(super_class_array), None

    def _get_non_separable_penalty(self, labels: np.ndarray):
        labels_hierarchy = self._get_hierarchy()
        distance_matrix, leaf_indexes = self._floyd_warshall(labels_hierarchy)
        distances = self._leaf_nodes_matrix(leaf_indexes, distance_matrix)
        penalties = distances[labels]
        return penalties, distances

    def _get_hierarchy(self):
        # Generate the tree
        root = utils.TreeNode("Root")
        graph = defaultdict(dict)
        graph = {"Root": []}

        diffs = np.where(self.first_level != 0)[0]
        change_indices = np.where(diffs != 0)[0]

        for i in range(1, 4):  # 3 Nodes at level 2
            level_2_node = utils.TreeNode(f"L2-{i}")
            root.children.append(level_2_node)
            graph["Root"].append(f"L2-{i}")
            graph[f"L2-{i}"] = []

            # Level 3
            if i == 1:
                level_3_limit_i = 0
                level_3_limit_f = change_indices[0] + 1
            elif i == 2:
                level_3_limit_i = change_indices[0] + 1
                level_3_limit_f = change_indices[1] + 1
            elif i == 3:
                level_3_limit_i = change_indices[1] + 1
                level_3_limit_f = 9
            else:
                raise ValueError("Invalid level 2 node")
            for j in range(level_3_limit_i, level_3_limit_f):  # Level 3 nodes
                level_3_node = utils.TreeNode(f"L3-{i}.{j}")
                level_2_node.children.append(level_3_node)
                graph[f"L2-{i}"].append(f"L3-{i}.{j}")
                graph[f"L3-{i}.{j}"] = []
                grouped = self.second_level[str(j)]
                for k in grouped:  # Level 4 nodes
                    leaf_node = utils.TreeNode(f"Leaf-{i}.{j}.{k}")
                    level_3_node.children.append(leaf_node)
                    graph[f"L3-{i}.{j}"].append(f"Leaf-{i}.{j}.{k}")
                    graph[f"Leaf-{i}.{j}.{k}"] = []

        return graph

    def _get_hierarchy_label(self, node_name: str) -> int:
        return int(node_name.split(".")[-1])


class FitzpatrickClean(Dataset):
    def __init__(
        self,
        hierarchy_artifacts: dict,
        labels: np.ndarray,
        rng: np.random.Generator,
        calibration_size: int = 3000,
    ):
        super().__init__(labels, rng, calibration_size)
        label_hierarchy = hierarchy_artifacts["label_hierarchy"]
        hierarchy_generator = hierarchies.FitzpatrickHierarchy(
            label_hierarchy=label_hierarchy, labels=labels
        )
        hierarchy = hierarchy_generator.get_hierarchy()
        self.first_level = hierarchy.first_level
        self.second_level = hierarchy.second_level

    def _get_non_separable_penalty(self, labels: np.ndarray):
        labels_hierarchy = self._fitzpatrick_hierarchy()
        distance_matrix, leaf_indexes = self._floyd_warshall(labels_hierarchy)
        distances = self._leaf_nodes_matrix(leaf_indexes, distance_matrix)
        penalties = distances[labels]
        return penalties, distances

    def _get_separable_penalty(self):
        # TODO: Check if this is the right formulation for the separable case
        bins = 4
        discrete_grid = np.array(list(range(1, bins)))

        labels_array = np.tile(np.arange(self.num_classes), (self.num_samples, 1))
        super_class_penalty = self.rng.choice(discrete_grid, size=10)
        penalties_array = np.vectorize(lambda label: super_class_penalty[label])
        super_class_array = labels_array // (self.num_classes // 10 + 1)
        return penalties_array(super_class_array), None
        raise NotImplementedError("Separable penalties not implemented for Fitzpatrick dataset")

    # TODO Refactor this code to make it more general
    def _fitzpatrick_hierarchy(self):
        # Level 2-0 = benign
        # Level 2-1 = malignant
        # Level 2-2 = non-neoplastic

        # Generate the tree
        root = utils.TreeNode("Root")
        graph = defaultdict(dict)
        graph = {"Root": []}

        diffs = np.where(self.first_level != 0)[0]
        change_indices = np.where(diffs != 0)[0]

        for i in range(1, 4):  # 3 Nodes at level 2
            level_2_node = utils.TreeNode(f"L2-{i}")
            root.children.append(level_2_node)
            graph["Root"].append(f"L2-{i}")
            graph[f"L2-{i}"] = []

            # Level 3

            if i == 1:
                level_3_limit_i = 0
                level_3_limit_f = change_indices[0] + 1
            elif i == 2:
                level_3_limit_i = change_indices[0] + 1
                level_3_limit_f = change_indices[1] + 1
            elif i == 3:
                level_3_limit_i = change_indices[1] + 1
                level_3_limit_f = 8
            else:
                raise ValueError("Invalid level 2 node")
            for j in range(level_3_limit_i, level_3_limit_f):  # Level 3 nodes
                level_3_node = utils.TreeNode(f"L3-{i}.{j}")
                level_2_node.children.append(level_3_node)
                graph[f"L2-{i}"].append(f"L3-{i}.{j}")
                graph[f"L3-{i}.{j}"] = []
                grouped = self.second_level[str(j)]
                for k in grouped:  # Level 4 nodes
                    leaf_node = utils.TreeNode(f"Leaf-{i}.{j}.{k}")
                    level_3_node.children.append(leaf_node)
                    graph[f"L3-{i}.{j}"].append(f"Leaf-{i}.{j}.{k}")
                    graph[f"Leaf-{i}.{j}.{k}"] = []

        return graph

    def _get_hierarchy_label(self, node_name):
        return int(node_name.split(".")[-1])


class Fitzpatrick(Dataset):
    def __init__(
        self,
        hierarchy_artifacts: dict,
        labels: np.ndarray,
        rng: np.random.Generator,
        calibration_size: int = 6_000,
    ):
        super().__init__(labels, rng, calibration_size)
        label_hierarchy = hierarchy_artifacts["label_hierarchy"]
        hierarchy_generator = hierarchies.FitzpatrickHierarchy(
            label_hierarchy=label_hierarchy, labels=labels
        )
        hierarchy = hierarchy_generator.get_hierarchy()
        self.first_level = hierarchy.first_level
        self.second_level = hierarchy.second_level

    def _get_non_separable_penalty(self, labels: np.ndarray):
        labels_hierarchy = self._fitzpatrick_hierarchy()
        distance_matrix, leaf_indexes = self._floyd_warshall(labels_hierarchy)
        distances = self._leaf_nodes_matrix(leaf_indexes, distance_matrix)
        penalties = distances[labels]
        return penalties, distances

    def _get_separable_penalty(self):
        # TODO: Check if this is the right formulation for the separable case
        bins = 4
        discrete_grid = np.array(list(range(1, bins)))

        labels_array = np.tile(np.arange(self.num_classes), (self.num_samples, 1))
        super_class_penalty = self.rng.choice(discrete_grid, size=10)
        penalties_array = np.vectorize(lambda label: super_class_penalty[label])
        super_class_array = labels_array // (self.num_classes // 10 + 1)
        return penalties_array(super_class_array), None
        raise NotImplementedError("Separable penalties not implemented for Fitzpatrick dataset")

    # TODO Refactor this code to make it more general
    def _fitzpatrick_hierarchy(self):
        # Level 2-0 = benign
        # Level 2-1 = malignant
        # Level 2-2 = non-neoplastic

        # Generate the tree
        root = utils.TreeNode("Root")
        graph = defaultdict(dict)
        graph = {"Root": []}

        diffs = np.where(self.first_level != 0)[0]
        change_indices = np.where(diffs != 0)[0]

        for i in range(1, 4):  # 3 Nodes at level 2
            level_2_node = utils.TreeNode(f"L2-{i}")
            root.children.append(level_2_node)
            graph["Root"].append(f"L2-{i}")
            graph[f"L2-{i}"] = []

            # Level 3

            if i == 1:
                level_3_limit_i = 0
                level_3_limit_f = change_indices[0] + 1
            elif i == 2:
                level_3_limit_i = change_indices[0] + 1
                level_3_limit_f = change_indices[1] + 1
            elif i == 3:
                level_3_limit_i = change_indices[1] + 1
                level_3_limit_f = 9
            else:
                raise ValueError("Invalid level 2 node")
            for j in range(level_3_limit_i, level_3_limit_f):  # Level 3 nodes
                level_3_node = utils.TreeNode(f"L3-{i}.{j}")
                level_2_node.children.append(level_3_node)
                graph[f"L2-{i}"].append(f"L3-{i}.{j}")
                graph[f"L3-{i}.{j}"] = []
                grouped = self.second_level[str(j)]
                for k in grouped:  # Level 4 nodes
                    leaf_node = utils.TreeNode(f"Leaf-{i}.{j}.{k}")
                    level_3_node.children.append(leaf_node)
                    graph[f"L3-{i}.{j}"].append(f"Leaf-{i}.{j}.{k}")
                    graph[f"Leaf-{i}.{j}.{k}"] = []

        return graph

    def _get_hierarchy_label(self, node_name):
        return int(node_name.split(".")[-1])


class iNaturalist(Dataset):
    def __init__(
        self,
        hierarchy_artifacts: dict,
        labels: np.ndarray,
        rng: np.random.Generator,
        calibration_size: int = 50_000,
    ):
        super().__init__(labels, rng, calibration_size)
        hierarchy, leaf_nodes = self._get_hierarchy()
        self.all_categories = hierarchy_artifacts["all_categories"]
        self.categories_index = hierarchy_artifacts["categories_index"]

        self.second_level = leaf_nodes

    def _get_non_separable_penalty(self, labels: np.ndarray):
        labels_hierarchy, _ = self._get_hierarchy()
        distance_matrix, leaf_indexes = self._floyd_warshall(labels_hierarchy)
        distances = self._leaf_nodes_matrix(leaf_indexes, distance_matrix)
        penalties = distances[labels]
        return penalties, distances

    def _get_separable_penalty(self):
        # TODO: Check if this is the right formulation for the separable case
        bins = 4
        discrete_grid = np.array(list(range(1, bins)))

        labels_array = np.tile(np.arange(self.num_classes), (self.num_samples, 1))
        super_class_penalty = self.rng.choice(discrete_grid, size=10)
        penalties_array = np.vectorize(lambda label: super_class_penalty[label])
        super_class_array = labels_array // (self.num_classes // 10 + 1)
        return penalties_array(super_class_array), None

    # TODO Refactor this code to make it more general, when the hiearchy is not
    # random is very easy to implement.
    def _get_hierarchy(self):
        # Generate the tree
        root = utils.TreeNode("Root")
        graph = {"Root": []}

        kingdoms = set()
        phyla = set()
        classes = set()
        # Construct the hierarchy up to the class level
        species_list = self.all_categories

        for species in species_list:
            parts = species.split('_')
            kingdom, phylum, class_ = parts[1], parts[2], parts[3]

            if kingdom not in kingdoms:
                kingdoms.add(kingdom)
                level_2_node = utils.TreeNode(f"L2-{kingdom}")
                root.children.append(level_2_node)
                graph["Root"].append(f"L2-{kingdom}")
                graph[f"L2-{kingdom}"] = []

            if phylum not in phyla:
                phyla.add(phylum)
                level_3_node = utils.TreeNode(f"L3-{kingdom}.{phylum}")
                graph[f"L2-{kingdom}"].append(f"L3-{kingdom}.{phylum}")
                graph[f"L3-{kingdom}.{phylum}"] = []

            if class_ not in classes:
                classes.add(class_)
                level_4_node = utils.TreeNode(f"Leaf-{kingdom}.{phylum}.{class_}")
                graph[f"L3-{kingdom}.{phylum}"].append(f"Leaf-{kingdom}.{phylum}.{class_}")
                graph[f"Leaf-{kingdom}.{phylum}.{class_}"] = []

        leaf_nodes = {}
        for key in graph.keys():
            if "Leaf" in key:
                label = self.categories_index["class"][key.split(".")[-1]]
                phylum = key.split(".")[-2]
                leaf_nodes.setdefault(phylum, []).append(label)

        return graph, leaf_nodes

    def _get_hierarchy_label(self, node_name):
        return self.categories_index["class"][node_name.split(".")[-1]]


class ImageNet(Dataset):
    def __init__(
        self,
        hierarchy_artifacts: dict,
        labels: np.ndarray,
        rng: np.random.Generator,
        calibration_size: int = 25_000,
    ):
        super().__init__(labels, rng, calibration_size)
        embeddings = hierarchy_artifacts["embeddings"]
        hierarchy_generator = hierarchies.ClusteringHierarchy(embeddings=embeddings, labels=labels)
        hierarchy = hierarchy_generator.get_hierarchy()
        self.first_level = hierarchy.first_level
        self.second_level = hierarchy.second_level

    def _get_non_separable_penalty(self, labels: np.ndarray):
        labels_hierarchy = self._get_hierarchy()
        distance_matrix, leaf_indexes = self._floyd_warshall(labels_hierarchy)
        distances = self._leaf_nodes_matrix(leaf_indexes, distance_matrix)
        penalties = distances[labels]
        return penalties, distances

    def _get_separable_penalty(self):
        # TODO: Check if this is the right formulation for the separable case
        bins = 4
        discrete_grid = np.array(list(range(1, bins)))

        labels_array = np.tile(np.arange(self.num_classes), (self.num_samples, 1))
        super_class_penalty = self.rng.choice(discrete_grid, size=10)
        penalties_array = np.vectorize(lambda label: super_class_penalty[label])
        super_class_array = labels_array // (self.num_classes // 10 + 1)
        return penalties_array(super_class_array), None

    # TODO Refactor this code to make it more general
    def _get_hierarchy(self):
        # Generate the tree
        root = utils.TreeNode("Root")
        graph = defaultdict(dict)
        graph = {"Root": []}

        diffs = np.where(self.first_level != 0)[0]
        change_indices = np.where(diffs != 0)[0]

        for i in range(1, 4):  # 3 Nodes at level 2
            level_2_node = utils.TreeNode(f"L2-{i}")
            root.children.append(level_2_node)
            graph["Root"].append(f"L2-{i}")
            graph[f"L2-{i}"] = []

            # Level 3

            if i == 1:
                level_3_limit_i = 0
                level_3_limit_f = change_indices[0] + 1
            elif i == 2:
                level_3_limit_i = change_indices[0] + 1
                level_3_limit_f = change_indices[1] + 1
            elif i == 3:
                level_3_limit_i = change_indices[1] + 1
                level_3_limit_f = 9
            else:
                raise ValueError("Invalid level 2 node")
            for j in range(level_3_limit_i, level_3_limit_f):  # Level 3 nodes
                level_3_node = utils.TreeNode(f"L3-{i}.{j}")
                level_2_node.children.append(level_3_node)
                graph[f"L2-{i}"].append(f"L3-{i}.{j}")
                graph[f"L3-{i}.{j}"] = []
                grouped = self.second_level[str(j)]
                for k in grouped:  # Level 4 nodes
                    leaf_node = utils.TreeNode(f"Leaf-{i}.{j}.{k}")
                    level_3_node.children.append(leaf_node)
                    graph[f"L3-{i}.{j}"].append(f"Leaf-{i}.{j}.{k}")
                    graph[f"Leaf-{i}.{j}.{k}"] = []

        return graph

    def _get_hierarchy_label(self, node_name):
        return int(node_name.split(".")[-1])
