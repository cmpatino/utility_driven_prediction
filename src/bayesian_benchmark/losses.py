from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from utils import compute_parallel_process, dict_to_list


@dataclass
class Evaluation:
    size: np.ndarray
    sum_penalties: np.ndarray = None
    max_penalties: np.ndarray = None
    coverage_loss: np.ndarray = None


class NeymanPearson:
    def __init__(self, calibration_size, leaf_nodes):
        self.is_separable = True
        self.is_lambda_independent = True
        self.calibration_size = calibration_size
        self.qhat = None

    def get_scores(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
        calibration_mode: bool = True,
        labels: np.ndarray = None,
    ) -> np.ndarray:
        if calibration_mode:
            return np.take_along_axis(probabilities / penalties, labels[:, None], axis=1)
        return probabilities / penalties

    def get_quantile(self, scores, alpha) -> float:
        q_level = np.ceil((self.calibration_size + 1) * alpha) / self.calibration_size
        self.qhat = np.quantile(scores, q_level, method="higher")
        return self.qhat

    def get_prediction_sets(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
    ):
        non_conf_score = self.get_scores(
            penalties=penalties,
            pairwise_distances=pairwise_distances,
            probabilities=probabilities,
            lambda_=lambda_,
            calibration_mode=False,
        )
        prediction_sets = non_conf_score >= self.qhat
        return prediction_sets

    def eval_sets(self, prediction_sets, penalties) -> Evaluation:
        size = np.sum(prediction_sets, axis=-1)
        losses_array = np.where(prediction_sets, penalties, 0)
        sum_penalties = np.sum(losses_array, axis=-1)
        max_penalties = np.max(losses_array, axis=-1)
        return Evaluation(size=size, sum_penalties=sum_penalties, max_penalties=max_penalties)


class Cumulative:
    def __init__(self, calibration_size, leaf_nodes):
        self.is_separable = True
        self.is_lambda_independent = False
        self.calibration_size = calibration_size
        self.qhat = None

    def get_scores(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
        calibration_mode: bool = True,
        labels: np.ndarray = None,
    ) -> np.ndarray:
        n_rows = probabilities.shape[0]
        sorted_prob_idxs = probabilities.argsort(1)[:, ::-1]
        srt_probs = np.take_along_axis(probabilities, sorted_prob_idxs, axis=1)

        srt_penalties = np.take_along_axis(penalties, sorted_prob_idxs, axis=1)
        if calibration_mode:
            # Get the position of the true label
            stop_proba_idx = np.where(sorted_prob_idxs == labels[:, None])[1]
            stop_penalty_idx = stop_proba_idx
        else:
            stop_proba_idx = ...
            stop_penalty_idx = ...

        base_scores = srt_probs.cumsum(axis=1)[np.arange(n_rows), stop_proba_idx].reshape(
            n_rows, -1
        )
        scores = srt_penalties.cumsum(axis=-1)[np.arange(n_rows), stop_penalty_idx].reshape(
            n_rows, -1
        )
        return base_scores + lambda_ * scores

    def get_quantile(self, scores, alpha) -> float:
        q_level = np.ceil((self.calibration_size + 1) * (1 - alpha)) / self.calibration_size
        self.qhat = np.quantile(scores, q_level, method="higher")
        return self.qhat

    def get_prediction_sets(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
    ):
        srt = probabilities.argsort(1)[:, ::-1]
        non_conf_score = self.get_scores(
            penalties=penalties,
            pairwise_distances=pairwise_distances,
            probabilities=probabilities,
            lambda_=lambda_,
            calibration_mode=False,
        )
        indicators = non_conf_score <= self.qhat
        prediction_sets = np.take_along_axis(indicators, srt.argsort(axis=1), axis=1)
        return prediction_sets

    def eval_sets(self, prediction_sets, penalties) -> Evaluation:
        size = np.sum(prediction_sets, axis=-1)
        losses_array = np.where(prediction_sets, penalties, 0)
        sum_penalties = np.sum(losses_array, axis=-1)
        max_penalties = np.max(losses_array, axis=-1)
        return Evaluation(size=size, sum_penalties=sum_penalties, max_penalties=max_penalties)


class MarginalMax:
    def __init__(self, calibration_size, leaf_nodes):
        self.is_separable = False
        self.is_lambda_independent = False
        self.calibration_size = calibration_size
        self.set_family = dict_to_list(leaf_nodes)

    def get_scores(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
        calibration_mode: bool = True,
        labels: np.ndarray = None,
    ) -> np.ndarray:
        n_rows = probabilities.shape[0]
        sorted_prob_idxs = probabilities.argsort(1)[:, ::-1]
        srt_probs = np.take_along_axis(probabilities, sorted_prob_idxs, axis=1)

        if calibration_mode:
            stop_proba_idx = np.where(sorted_prob_idxs == labels[:, None])[1]
            stop_penalty_idx = stop_proba_idx
        else:
            stop_proba_idx = ...
            stop_penalty_idx = ...

        base_score = srt_probs.cumsum(axis=1)[np.arange(n_rows), stop_proba_idx].reshape(n_rows, -1)

        penalties_by_softmax = np.take_along_axis(penalties, sorted_prob_idxs, axis=1)
        cum_max = np.maximum.accumulate(penalties_by_softmax, axis=1)
        differences = np.diff(cum_max, axis=1, prepend=0)
        scores = np.cumsum(differences, axis=1)[np.arange(n_rows), stop_penalty_idx].reshape(
            n_rows, -1
        )

        return base_score + lambda_ * scores

    def get_quantile(self, scores, alpha) -> float:
        q_level = np.ceil((self.calibration_size + 1) * (1 - alpha)) / self.calibration_size
        self.qhat = np.quantile(scores, q_level, method="higher")
        return self.qhat

    def get_prediction_sets(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
    ):
        srt = probabilities.argsort(1)[:, ::-1]
        non_conf_score = self.get_scores(
            penalties=penalties,
            pairwise_distances=pairwise_distances,
            probabilities=probabilities,
            lambda_=lambda_,
            calibration_mode=False,
        )
        indicators = non_conf_score <= self.qhat
        prediction_sets = np.take_along_axis(indicators, srt.argsort(axis=1), axis=1)
        return prediction_sets

    def _compute_coverage_function(self, label_set):
        value = 0
        for subset in self.set_family:
            value += 1 if bool(set(subset) & set(label_set)) else 0

        return value

    def eval_sets(self, prediction_sets, penalties):
        label_sets = []
        labels = np.arange(prediction_sets.shape[1])
        values = []
        size = np.sum(prediction_sets, axis=-1)
        for i in range(prediction_sets.shape[0]):
            indices = np.where(prediction_sets[i])
            l_set = labels[indices]
            label_sets.append(l_set)
            values.append(self._compute_coverage_function(l_set))

        coverage_loss = np.array(values)

        losses_array = np.where(prediction_sets, penalties, 0)
        max_penalties = np.max(losses_array, axis=-1)
        sum_penalties = np.sum(losses_array, axis=-1)
        return Evaluation(
            size=size,
            coverage_loss=coverage_loss,
            max_penalties=max_penalties,
            sum_penalties=sum_penalties,
        )


class SubmodularGreedy(ABC):
    def __init__(self, calibration_size, quantile=None):
        self.is_separable = False
        self.is_lambda_independent = True
        self.calibration_size = calibration_size
        self.quantile = quantile
        self.distances = None

    def compute_non_conformity_score(self, calibration_data_point):
        predictions, label = calibration_data_point
        curr_p_sum = 0
        prediction_set = np.zeros_like(predictions)
        current_max_distances = {}
        max_idx = self._compute_greedy_step(predictions, prediction_set, current_max_distances)
        prediction_set[max_idx] = 1
        curr_p_sum += predictions[max_idx]

        index_order = []
        index_order.append((max_idx, predictions[max_idx], 0))
        # Dictionary to keep track of maximum distances for current set

        index = 0
        if max_idx != label:
            while np.sum(prediction_set) < len(predictions):
                index += 1
                max_idx = self._compute_greedy_step(
                    predictions, prediction_set, current_max_distances
                )

                prediction_set[max_idx] = 1
                curr_p_sum += predictions[max_idx]
                if max_idx == label:
                    break

        index_order.append((max_idx, predictions[max_idx], index))
        return curr_p_sum

    def compute_prediction_sets(self, predictions):
        curr_p_sum = 0
        prediction_set = np.zeros_like(predictions)
        current_max_distances = {}
        max_idx = self._compute_greedy_step(predictions, prediction_set, current_max_distances)
        prediction_set[max_idx] = 1
        curr_p_sum += predictions[max_idx]
        while curr_p_sum < self.quantile:
            max_idx = self._compute_greedy_step(predictions, prediction_set, current_max_distances)
            prediction_set[max_idx] = 1
            curr_p_sum += predictions[max_idx]

        prediction_set[max_idx] = 0
        return prediction_set.astype(bool)

    def get_scores(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
        calibration_mode: bool = True,
        labels: np.ndarray = None,
    ) -> np.ndarray:
        self.distances = pairwise_distances
        data = []
        for index in range(len(probabilities)):
            predictions = probabilities[index]
            label = labels[index]
            data.append((predictions, label))

        nc_scores = compute_parallel_process(self.compute_non_conformity_score, data, True)

        base_scores = np.array(nc_scores)
        return base_scores[:, None]

    def get_quantile(self, scores, alpha) -> float:
        q_level = np.ceil((self.calibration_size + 1) * (1 - alpha)) / self.calibration_size
        self.quantile = np.quantile(scores, q_level, method="higher")
        return self.quantile

    def get_prediction_sets(
        self,
        penalties: np.ndarray,
        pairwise_distances: np.ndarray,
        probabilities: np.ndarray,
        lambda_: float,
    ):
        predictions_set = compute_parallel_process(
            self.compute_prediction_sets, probabilities, True
        )
        predictions_set = np.array(predictions_set)
        return predictions_set

    def eval_sets(self, prediction_sets, penalties):
        label_sets = []
        labels = np.arange(prediction_sets.shape[1])
        values = []
        size = np.sum(prediction_sets, axis=-1)
        for i in range(prediction_sets.shape[0]):
            indices = np.where(prediction_sets[i])
            l_set = labels[indices]
            label_sets.append(l_set)
            values.append(self._compute_coverage_function(l_set))

        coverage_loss = np.array(values)

        losses_array = np.where(prediction_sets, penalties, 0)
        max_penalties = np.max(losses_array, axis=-1)
        sum_penalties = np.sum(losses_array, axis=-1)
        return Evaluation(
            size=size,
            coverage_loss=coverage_loss,
            max_penalties=max_penalties,
            sum_penalties=sum_penalties,
        )

    @abstractmethod
    def _compute_greedy_step(self, predictions, prediction_set, current_max_distances):
        raise NotImplementedError()


class GreedyCoverageFunction(SubmodularGreedy):
    def __init__(self, calibration_size, leaf_nodes, quantile=None):
        super().__init__(calibration_size, quantile)

        leaf_nodes = leaf_nodes
        self.set_family = dict_to_list(leaf_nodes)

    def _compute_coverage_function(self, label_set):
        value = 0
        for subset in self.set_family:
            value += 1 if bool(set(subset) & set(label_set)) else 0

        return value

    def _compute_greedy_step(self, predictions, prediction_set, current_max_distances):
        max_idx = None
        max_ratio = float("-inf")
        current_set_indexes = np.arange(len(predictions))[prediction_set == 1].tolist()
        diameter = len(self.set_family)
        for index, indicator in enumerate(prediction_set):
            if indicator == 1:
                continue

            candidate_set = current_set_indexes + [index]

            new_value = self._compute_coverage_function(candidate_set)

            inner_curr_ratio = (diameter - new_value) / ((1 - predictions[index]) + 1e-6)

            if inner_curr_ratio > max_ratio:
                max_ratio = inner_curr_ratio
                max_idx = index
        return max_idx


class GreedyMax(SubmodularGreedy):
    def __init__(self, calibration_size, leaf_nodes, quantile=None):
        super().__init__(calibration_size, quantile)

        leaf_nodes = leaf_nodes
        self.set_family = dict_to_list(leaf_nodes)

    def _compute_coverage_function(self, label_set):
        value = 0
        for subset in self.set_family:
            value += 1 if bool(set(subset) & set(label_set)) else 0

        return value

    def _compute_greedy_step(self, predictions, prediction_set, current_max_distances):
        max_idx = None
        max_ratio = float("-inf")
        diameter = np.max(self.distances) + 1
        for index, indicator in enumerate(prediction_set):
            if indicator == 1:
                continue
            current_set_indexes = np.arange(len(predictions))[prediction_set == 1].tolist()

            new_maximum = 0
            for existing_idx in current_set_indexes:
                # Standardizing the pair (min, max)
                pair = (min(existing_idx, index), max(existing_idx, index))
                if pair not in current_max_distances:
                    current_max_distances[pair] = self.distances[pair[0], pair[1]]

                new_maximum = max(new_maximum, current_max_distances[pair])

            inner_curr_ratio = (diameter + 1 - new_maximum) / (1 - predictions[index] + 1e-6)

            if inner_curr_ratio > max_ratio:
                max_ratio = inner_curr_ratio
                max_idx = index

        return max_idx
