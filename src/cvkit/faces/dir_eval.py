import pickle

import numpy as np

from .neighbors import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        train_embeddings = np.asarray(self.train_embeddings)
        train_labels = np.asarray(self.train_labels)
        test_embeddings = np.asarray(self.test_embeddings)

        valid_train = np.all(np.isfinite(train_embeddings), axis=1) & np.isfinite(train_labels)
        if not np.any(valid_train):
            similarity_thresholds = np.full_like(
                self.false_alarm_rate_range, fill_value=0.0, dtype=float
            )
            identification_rates = np.zeros_like(self.false_alarm_rate_range, dtype=float)
            return {
                "false_alarm_rates": self.false_alarm_rate_range,
                "similarity_thresholds": similarity_thresholds,
                "identification_rates": identification_rates,
            }

        self.classifier.fit(train_embeddings[valid_train], train_labels[valid_train])
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(
            test_embeddings
        )

        similarity_thresholds = []
        identification_rates = []
        for false_alarm_rate in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(similarities, false_alarm_rate)
            thresholded_predictions = np.asarray(prediction_labels).copy()
            thresholded_predictions[np.asarray(similarities) < threshold] = UNKNOWN_LABEL

            similarity_thresholds.append(threshold)
            identification_rates.append(self.calc_identification_rate(thresholded_predictions))

        similarity_thresholds = np.asarray(similarity_thresholds, dtype=float)
        identification_rates = np.asarray(identification_rates, dtype=float)

        # Report all performance measures.
        evaluation_results = {
            "false_alarm_rates": self.false_alarm_rate_range,
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        similarity = np.asarray(similarity, dtype=float)
        labels = np.asarray(self.test_labels)
        unknown_mask = labels == UNKNOWN_LABEL
        unknown_similarity = similarity[unknown_mask & np.isfinite(similarity)]

        if len(unknown_similarity) == 0:
            finite_similarity = similarity[np.isfinite(similarity)]
            if len(finite_similarity) == 0:
                return 0.0
            return float(np.nextafter(np.max(finite_similarity), np.inf))

        false_alarm_rate = float(np.clip(false_alarm_rate, 0.0, 1.0))
        if false_alarm_rate <= 0.0:
            return float(np.nextafter(np.max(unknown_similarity), np.inf))
        if false_alarm_rate >= 1.0:
            return float(np.min(unknown_similarity))

        percentile = (1.0 - false_alarm_rate) * 100.0
        return float(np.percentile(unknown_similarity, percentile))

    def calc_identification_rate(self, prediction_labels):
        prediction_labels = np.asarray(prediction_labels)
        true_labels = np.asarray(self.test_labels)
        known_mask = true_labels != UNKNOWN_LABEL
        if not np.any(known_mask):
            return 0.0
        correct = prediction_labels[known_mask] == true_labels[known_mask]
        return float(np.count_nonzero(correct) / np.count_nonzero(known_mask))
