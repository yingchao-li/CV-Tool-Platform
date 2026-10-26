from collections.abc import Callable
from typing import Final, Optional
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import ConvergenceWarning

from .config import Config

UNKNOWN_LABEL: Final[int] = -1
RANDOM_STATE: Final[int] = 42
SCORE_FLOOR: Final[float] = -1.0e12


class _FeaturePreprocessor:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_: Optional[int] = None

    def fit(self, x: np.ndarray) -> "_FeaturePreprocessor":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Training features must be a non-empty 2D array.")

        self.n_features_ = x.shape[1]
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.n_features_ is None:
            raise RuntimeError("Feature preprocessor has not been fitted.")

        x = _as_feature_matrix(x, self.n_features_)
        if len(x) == 0:
            return x.astype(float)

        x = x.astype(float, copy=True)
        bad = ~np.isfinite(x)
        if np.any(bad):
            x[bad] = np.take(self.mean_, np.where(bad)[1])

        return (x - self.mean_) / self.scale_


class _OpenSetLinearModel:
    def __init__(
        self,
        preprocessor: _FeaturePreprocessor,
        classifier: SGDClassifier,
        known_labels: np.ndarray,
        pseudo_labels: np.ndarray,
        train_features: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.known_labels = np.asarray(known_labels, dtype=int)
        self.pseudo_labels = np.asarray(pseudo_labels, dtype=int)
        self.train_features = np.asarray(train_features, dtype=np.float32)
        self.train_labels = np.asarray(train_labels, dtype=int)

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test = self.preprocessor.transform(x_test)
        n_samples = len(x_test)
        if n_samples == 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=float)

        scores = _decision_scores(self.classifier, x_test)
        classes = np.asarray(self.classifier.classes_, dtype=int)
        known_columns = np.flatnonzero(np.isin(classes, self.known_labels))
        pseudo_columns = np.flatnonzero(np.isin(classes, self.pseudo_labels))
        if len(known_columns) == 0 or len(pseudo_columns) == 0:
            raise RuntimeError("Classifier is missing known or pseudo-class score columns.")

        known_scores = scores[:, known_columns]
        pseudo_scores = scores[:, pseudo_columns]
        best_known = np.argmax(known_scores, axis=1)
        max_known_scores = known_scores[np.arange(n_samples), best_known]
        max_pseudo_scores = np.max(pseudo_scores, axis=1)

        distance_labels, y_score = self._nearest_distance_margin(x_test)
        decision_accepts_known = max_known_scores >= max_pseudo_scores
        y_score = np.nan_to_num(y_score, nan=SCORE_FLOOR, neginf=SCORE_FLOOR, posinf=-SCORE_FLOOR)

        y_pred = classes[known_columns][best_known].astype(int)
        use_distance_label = distance_labels != UNKNOWN_LABEL
        y_pred[use_distance_label] = distance_labels[use_distance_label]
        y_pred[(y_score < 0.0) | ~decision_accepts_known] = UNKNOWN_LABEL
        return y_pred.reshape(-1), y_score.astype(float).reshape(-1)

    def _nearest_distance_margin(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        known_mask = np.isin(self.train_labels, self.known_labels)
        pseudo_mask = np.isin(self.train_labels, self.pseudo_labels)
        known_features = self.train_features[known_mask]
        known_labels = self.train_labels[known_mask]
        pseudo_features = self.train_features[pseudo_mask]

        nearest_known_labels = np.full((len(x_test),), UNKNOWN_LABEL, dtype=int)
        margins = np.full((len(x_test),), SCORE_FLOOR, dtype=float)
        if len(known_features) == 0 or len(pseudo_features) == 0:
            return nearest_known_labels, margins

        batch_size = 512
        for start in range(0, len(x_test), batch_size):
            end = start + batch_size
            batch = x_test[start:end].astype(np.float64)
            known_dist = _squared_distances(batch, known_features)
            pseudo_dist = _squared_distances(batch, pseudo_features)

            nearest_known = np.argmin(known_dist, axis=1)
            nearest_known_distance = np.sqrt(known_dist[np.arange(len(batch)), nearest_known])
            nearest_pseudo_distance = np.sqrt(np.min(pseudo_dist, axis=1))

            nearest_known_labels[start:end] = known_labels[nearest_known]
            margins[start:end] = nearest_pseudo_distance - nearest_known_distance

        return nearest_known_labels, margins


def _as_feature_matrix(x: np.ndarray, n_features: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        if n_features is None:
            return np.empty((0, 0), dtype=float)
        return np.empty((0, n_features), dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("Feature array must be 1D or 2D.")
    if n_features is not None and x.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, got {x.shape[1]}.")
    return x


def _clean_training_data(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = _as_feature_matrix(x_train)
    y_train = np.asarray(y_train).reshape(-1)
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must contain the same number of samples.")
    if len(x_train) == 0:
        raise ValueError("Training data is empty.")

    finite_rows = np.all(np.isfinite(x_train), axis=1) & np.isfinite(y_train.astype(float))
    x_train = x_train[finite_rows]
    y_train = y_train[finite_rows].astype(int)
    if len(x_train) == 0:
        raise ValueError("Training data contains no finite samples.")
    if not np.any(y_train >= 0):
        raise ValueError("Training data must contain at least one known class.")
    if not np.any(y_train == UNKNOWN_LABEL):
        raise ValueError("Training data must contain at least one known unknown sample.")
    return x_train, y_train


def _make_classifier() -> SGDClassifier:
    return SGDClassifier(
        loss="modified_huber",
        alpha=1.0e-4,
        max_iter=4000,
        tol=1.0e-2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _fit_open_set_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    pseudo_labels_for_unknown: np.ndarray,
) -> _OpenSetLinearModel:
    x_train, y_train = _clean_training_data(x_train, y_train)
    unknown_mask = y_train == UNKNOWN_LABEL
    if len(pseudo_labels_for_unknown) != np.count_nonzero(unknown_mask):
        raise ValueError("Pseudo-label count does not match the number of unknown samples.")

    known_labels = np.unique(y_train[y_train >= 0]).astype(int)
    pseudo_labels = np.unique(pseudo_labels_for_unknown).astype(int)

    mapped_labels = y_train.copy()
    mapped_labels[unknown_mask] = pseudo_labels_for_unknown.astype(int)

    preprocessor = _FeaturePreprocessor().fit(x_train)
    x_train = preprocessor.transform(x_train)

    classifier = _make_classifier()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(x_train, mapped_labels)
    return _OpenSetLinearModel(
        preprocessor,
        classifier,
        known_labels,
        pseudo_labels,
        x_train,
        mapped_labels,
    )


def _pseudo_label_start(y_train: np.ndarray) -> int:
    known_labels = np.asarray(y_train)[np.asarray(y_train) >= 0]
    if len(known_labels) == 0:
        raise ValueError("Training data must contain at least one known class.")
    return int(np.max(known_labels)) + 1


def _decision_scores(classifier: SGDClassifier, x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_scores = classifier.decision_function(x)
    raw_scores = np.asarray(raw_scores, dtype=float)
    classes = np.asarray(classifier.classes_)

    if raw_scores.ndim == 1:
        if len(classes) != 2:
            raw_scores = raw_scores.reshape(-1, 1)
        else:
            raw_scores = np.column_stack((-raw_scores, raw_scores))
    scores = np.nan_to_num(raw_scores, nan=SCORE_FLOOR, neginf=SCORE_FLOOR, posinf=-SCORE_FLOOR)
    return scores


def _squared_distances(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    x_norm = np.sum(x**2, axis=1, keepdims=True)
    reference_norm = np.sum(reference**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        distances = x_norm + reference_norm[None, :] - 2.0 * (x @ reference.T)
    distances = np.nan_to_num(distances, nan=np.inf, posinf=np.inf)
    return np.maximum(distances, 0.0)


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    x_train, y_train = _clean_training_data(x_train, y_train)
    pseudo_label = _pseudo_label_start(y_train)
    pseudo_labels = np.full(np.count_nonzero(y_train == UNKNOWN_LABEL), pseudo_label, dtype=int)
    model = _fit_open_set_model(x_train, y_train, pseudo_labels)

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return model.predict(x_test)

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    x_train, y_train = _clean_training_data(x_train, y_train)
    pseudo_start = _pseudo_label_start(y_train)
    pseudo_labels = pseudo_start + np.arange(np.count_nonzero(y_train == UNKNOWN_LABEL))
    model = _fit_open_set_model(x_train, y_train, pseudo_labels)

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return model.predict(x_test)

    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()

    print("Challenge data: {} samples, {} features".format(x_train.shape[0], x_train.shape[1]))
    print("Known samples: {}".format(np.count_nonzero(y_train >= 0)))
    print("Known unknown samples: {}".format(np.count_nonzero(y_train == UNKNOWN_LABEL)))

    for name, training_fn in (("SPL", spl_training), ("MPL", mpl_training)):
        start = time.perf_counter()
        predict_fn = training_fn(x_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y_pred, y_score = predict_fn(x_train)
        predict_seconds = time.perf_counter() - start

        print(
            "{}: train={:.3f}s predict={:.3f}s y_pred={} y_score={} unknown={} finite_scores={}".format(
                name,
                train_seconds,
                predict_seconds,
                y_pred.shape,
                y_score.shape,
                np.count_nonzero(y_pred == UNKNOWN_LABEL),
                np.all(np.isfinite(y_score)),
            )
        )


if __name__ == "__main__":
    main()
