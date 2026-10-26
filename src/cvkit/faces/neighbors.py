import cv2
import numpy as np


class NearestNeighborClassifier:

    def __init__(self):
        self.use_opencv_ml = hasattr(cv2, "ml") and hasattr(cv2.ml, "KNearest_create")
        self.classifier = cv2.ml.KNearest_create() if self.use_opencv_ml else None
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.float32)
        self.__reset()

    def __reset(self):
        if self.use_opencv_ml:
            self.classifier.setDefaultK(1)

    def fit(self, embeddings, labels):
        self.__reset()
        embeddings = embeddings.astype(np.float32)
        labels = labels.astype(np.float32)
        if self.use_opencv_ml:
            self.classifier.train(embeddings, cv2.ml.ROW_SAMPLE, labels)
            return

        self.embeddings = embeddings
        self.labels = labels

    def predict_labels_and_similarities(self, embeddings):
        embeddings = embeddings.astype(np.float32)
        if self.use_opencv_ml:
            _, prediction_labels, _, dists = self.classifier.findNearest(embeddings, k=1)
            similarities = -dists.flatten()
            return prediction_labels.flatten(), similarities

        prediction_labels = np.empty((len(embeddings),), dtype=np.float32)
        similarities = np.empty((len(embeddings),), dtype=np.float32)
        train_embeddings = self.embeddings.astype(np.float64)
        train_norms = np.sum(train_embeddings**2, axis=1)
        batch_size = 512
        for start in range(0, len(embeddings), batch_size):
            end = start + batch_size
            batch = embeddings[start:end].astype(np.float64)
            batch_norms = np.sum(batch**2, axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                squared_dists = batch_norms + train_norms[None, :] - 2.0 * (
                    batch @ train_embeddings.T
                )
            squared_dists = np.nan_to_num(squared_dists, nan=np.inf, posinf=np.inf)
            squared_dists = np.maximum(squared_dists, 0.0)
            nearest = np.argmin(squared_dists, axis=1)
            prediction_labels[start:end] = self.labels[nearest]
            similarities[start:end] = -np.sqrt(squared_dists[np.arange(len(batch)), nearest])
        return prediction_labels.flatten(), similarities
