"""Task-region diagnostics for frozen visual features and RSSM states."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt

import numpy as np


def _validate_task_arrays(
    task_arrays: Mapping[str, np.ndarray],
) -> tuple[list[str], int]:
    names = list(task_arrays)
    if len(names) < 2:
        raise ValueError("Task-region analysis requires at least two tasks")
    feature_dim = -1
    for name in names:
        values = np.asarray(task_arrays[name])
        if values.ndim != 2 or values.shape[0] < 4:
            raise ValueError(
                f"Task {name!r} must provide at least four [sample, feature] rows"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Task {name!r} contains non-finite features")
        if feature_dim < 0:
            feature_dim = values.shape[1]
        elif values.shape[1] != feature_dim:
            raise ValueError("All task feature widths must match")
    return names, feature_dim


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 1:
        raise ValueError("Wilson interval requires at least one observation")
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    distances = (
        np.square(first).sum(axis=1, keepdims=True)
        + np.square(second).sum(axis=1)[None, :]
        - 2.0 * first @ second.T
    )
    return np.maximum(distances, 0.0)


def _nearest_centroid_predictions(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
    class_count: int,
) -> np.ndarray:
    centroids = np.stack(
        [train[train_labels == task].mean(axis=0) for task in range(class_count)]
    )
    return _squared_distances(test, centroids).argmin(axis=1)


def _knn_predictions(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
    *,
    class_count: int,
    neighbors: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if neighbors < 1 or neighbors > len(train):
        raise ValueError("neighbors must fit within the training split")
    predictions = []
    for start in range(0, len(test), chunk_size):
        distances = _squared_distances(test[start : start + chunk_size], train)
        indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
        neighbor_labels = train_labels[indices]
        votes = np.stack(
            [(neighbor_labels == task).sum(axis=1) for task in range(class_count)],
            axis=1,
        )
        predictions.append(votes.argmax(axis=1))
    return np.concatenate(predictions)


def analyze_task_regions(
    task_arrays: Mapping[str, np.ndarray],
    *,
    seed: int = 0,
    max_samples_per_task: int = 512,
    test_fraction: float = 0.5,
    neighbors: int = 5,
    distance_projection_dim: int = 128,
    permutation_repetitions: int = 200,
) -> dict[str, object]:
    """Measure whether a held-out representation makes task identity decodable.

    Task labels are used only by this offline evaluator. They are never passed
    to the model or policy.
    """
    names, feature_dim = _validate_task_arrays(task_arrays)
    if max_samples_per_task < 4:
        raise ValueError("max_samples_per_task must be at least four")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if distance_projection_dim < 1:
        raise ValueError("distance_projection_dim must be positive")
    if permutation_repetitions < 1:
        raise ValueError("permutation_repetitions must be positive")

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    train_labels = []
    test_labels = []
    samples_per_task: dict[str, int] = {}
    balanced_count = min(
        max_samples_per_task,
        *(len(np.asarray(task_arrays[name])) for name in names),
    )
    for task_index, name in enumerate(names):
        values = np.asarray(task_arrays[name], dtype=np.float64)
        count = balanced_count
        chosen = rng.choice(len(values), size=count, replace=False)
        chosen_values = values[chosen]
        test_count = min(count - 2, max(2, int(round(count * test_fraction))))
        permutation = rng.permutation(count)
        test_indices = permutation[:test_count]
        train_indices = permutation[test_count:]
        train_parts.append(chosen_values[train_indices])
        test_parts.append(chosen_values[test_indices])
        train_labels.append(np.full(len(train_indices), task_index, dtype=np.int64))
        test_labels.append(np.full(len(test_indices), task_index, dtype=np.int64))
        samples_per_task[name] = count

    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    y_train = np.concatenate(train_labels)
    y_test = np.concatenate(test_labels)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    informative = scale > 1e-8
    if not informative.any():
        raise ValueError("Representation is constant across the training split")
    train = (train[:, informative] - mean[informative]) / scale[informative]
    test = (test[:, informative] - mean[informative]) / scale[informative]

    class_count = len(names)
    nearest_predictions = _nearest_centroid_predictions(
        train, y_train, test, class_count
    )
    nearest_successes = int((nearest_predictions == y_test).sum())
    nearest_accuracy = nearest_successes / len(y_test)

    projected_dim = min(distance_projection_dim, train.shape[1])
    if projected_dim < train.shape[1]:
        projection = rng.normal(
            0.0,
            1.0 / sqrt(projected_dim),
            size=(train.shape[1], projected_dim),
        )
        knn_train = train @ projection
        knn_test = test @ projection
    else:
        knn_train = train
        knn_test = test
    effective_neighbors = min(neighbors, len(knn_train))
    knn_predictions = _knn_predictions(
        knn_train,
        y_train,
        knn_test,
        class_count=class_count,
        neighbors=effective_neighbors,
    )
    knn_successes = int((knn_predictions == y_test).sum())
    knn_accuracy = knn_successes / len(y_test)

    centroids = np.stack([train[y_train == task].mean(0) for task in range(class_count)])
    within_mean_square = np.asarray(
        [
            np.square(train[y_train == task] - centroids[task]).sum(axis=1).mean()
            for task in range(class_count)
        ]
    )
    centroid_distances = np.sqrt(_squared_distances(centroids, centroids))
    normalized_distances = np.zeros_like(centroid_distances)
    for first in range(class_count):
        for second in range(first + 1, class_count):
            denominator = sqrt(
                max(0.5 * (within_mean_square[first] + within_mean_square[second]), 1e-12)
            )
            value = centroid_distances[first, second] / denominator
            normalized_distances[first, second] = value
            normalized_distances[second, first] = value
    off_diagonal = normalized_distances[np.triu_indices(class_count, 1)]

    null_accuracies = []
    for _ in range(permutation_repetitions):
        permuted_train = rng.permutation(y_train)
        permuted_test = rng.permutation(y_test)
        predictions = _nearest_centroid_predictions(
            train,
            permuted_train,
            test,
            class_count,
        )
        null_accuracies.append(float((predictions == permuted_test).mean()))
    permutation_p = (
        1
        + sum(value >= nearest_accuracy for value in null_accuracies)
    ) / (permutation_repetitions + 1)

    nearest_interval = _wilson_interval(nearest_successes, len(y_test))
    knn_interval = _wilson_interval(knn_successes, len(y_test))
    chance = 1.0 / class_count
    separation_detected = bool(
        nearest_interval[0] > chance
        and knn_interval[0] > chance
        and permutation_p <= 0.01
    )
    return {
        "task_names": names,
        "task_count": class_count,
        "original_feature_dim": feature_dim,
        "informative_feature_dim": int(informative.sum()),
        "samples_per_task": samples_per_task,
        "train_samples": len(train),
        "test_samples": len(test),
        "chance_accuracy": chance,
        "nearest_centroid": {
            "accuracy": nearest_accuracy,
            "wilson_95": list(nearest_interval),
            "permutation_p_value": permutation_p,
            "permutation_repetitions": permutation_repetitions,
        },
        "knn": {
            "neighbors": effective_neighbors,
            "accuracy": knn_accuracy,
            "wilson_95": list(knn_interval),
            "random_projection_dim": projected_dim,
        },
        "normalized_centroid_distance": {
            "definition": "centroid distance divided by pooled within-task RMS radius",
            "matrix": normalized_distances.tolist(),
            "mean_off_diagonal": float(off_diagonal.mean()),
            "minimum_off_diagonal": float(off_diagonal.min()),
        },
        "region_separation_detected": separation_detected,
        "interpretation": (
            "Held-out task identity is decodable above chance; this supports distinct "
            "task regions but does not prove disjoint supports or causal retention."
            if separation_detected
            else "This diagnostic does not establish reliable task-region separation."
        ),
    }


def task_support_overlap(
    task_support: Mapping[str, np.ndarray],
    *,
    epsilon: float = 1e-12,
) -> dict[str, object]:
    """Compute pairwise weighted-Jaccard overlap of mean RBF supports."""
    names = list(task_support)
    if len(names) < 2:
        raise ValueError("RBF support overlap requires at least two tasks")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    supports = [np.asarray(task_support[name], dtype=np.float64).reshape(-1) for name in names]
    if not supports[0].size or any(support.shape != supports[0].shape for support in supports):
        raise ValueError("All task RBF supports must have one matching non-empty shape")
    if any(not np.isfinite(support).all() for support in supports):
        raise ValueError("RBF supports must be finite")
    if any((support < 0).any() for support in supports):
        raise ValueError("RBF supports must be non-negative")
    matrix = np.eye(len(names), dtype=np.float64)
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            intersection = np.minimum(supports[first], supports[second]).sum()
            union = max(np.maximum(supports[first], supports[second]).sum(), epsilon)
            matrix[first, second] = intersection / union
            matrix[second, first] = matrix[first, second]
    off_diagonal = matrix[np.triu_indices(len(names), 1)]
    return {
        "task_names": names,
        "weighted_jaccard_matrix": matrix.tolist(),
        "mean_off_diagonal": float(off_diagonal.mean()),
        "minimum_off_diagonal": float(off_diagonal.min()),
        "maximum_off_diagonal": float(off_diagonal.max()),
    }
