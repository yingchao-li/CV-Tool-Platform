import os
import pickle

import cv2
import numpy as np

from .config import Config

UNKNOWN_LABEL = "unknown"


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = face.astype(np.float32)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - np.array(
            (131.0912, 103.8827, 91.4953), dtype=np.float32
        )

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm == 0.0:
            return embedding
        return embedding / norm

    @classmethod
    @property
    def embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:
    DEFAULT_NUM_NEIGHBOURS = 5
    DEFAULT_MAX_DISTANCE = 1.05
    DEFAULT_MIN_PROB = 0.5

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=1, max_distance=2.0, min_prob=0.0):
        self.facenet = FaceNet()
        if num_neighbours == 1 and max_distance == 2.0 and min_prob == 0.0:
            num_neighbours = self.DEFAULT_NUM_NEIGHBOURS
            max_distance = self.DEFAULT_MAX_DISTANCE
            min_prob = self.DEFAULT_MIN_PROB
        self.num_neighbours = max(1, int(num_neighbours))
        self.max_distance = float(max_distance)
        self.min_prob = float(min_prob)
        self.unknown_label = UNKNOWN_LABEL

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)
        self.labels = list(self.labels)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32).reshape(
            -1, FaceNet.embedding_dimensionality
        )

    def partial_fit(self, face, label):
        embeddings = self._extract_color_and_gray_embeddings(face)
        if not embeddings:
            return None

        new_embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
        self.labels.extend([label] * len(new_embeddings))
        return None

    def predict(self, face) -> tuple[str, float, float]:
        if len(self.labels) == 0 or self.embeddings.size == 0:
            return self.unknown_label, 0.0, float("inf")

        query_embeddings = self._extract_color_and_gray_embeddings(face)
        if not query_embeddings:
            return self.unknown_label, 0.0, float("inf")

        gallery_embeddings = np.asarray(self.embeddings, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=object)
        valid_gallery = np.all(np.isfinite(gallery_embeddings), axis=1)
        if not np.any(valid_gallery):
            return self.unknown_label, 0.0, float("inf")

        gallery_embeddings = gallery_embeddings[valid_gallery]
        labels = labels[valid_gallery]

        distances = []
        distance_labels = []
        for query_embedding in query_embeddings:
            diff = gallery_embeddings - query_embedding
            query_distances = np.linalg.norm(diff, axis=1)
            valid_distances = np.isfinite(query_distances)
            distances.extend(query_distances[valid_distances])
            distance_labels.extend(labels[valid_distances])

        if not distances:
            return self.unknown_label, 0.0, float("inf")

        distances = np.asarray(distances, dtype=np.float32)
        distance_labels = np.asarray(distance_labels, dtype=object)
        k = min(self.num_neighbours, len(distances))
        nearest_indices = np.argsort(distances, kind="mergesort")[:k]
        nearest_labels = distance_labels[nearest_indices]
        nearest_distances = distances[nearest_indices]

        predicted_label = self._majority_vote(nearest_labels, nearest_distances)
        prediction_mask = nearest_labels == predicted_label
        posterior = float(np.count_nonzero(prediction_mask) / k)
        prediction_distance = float(np.min(nearest_distances[prediction_mask]))

        if prediction_distance > self.max_distance or posterior < self.min_prob:
            return self.unknown_label, posterior, prediction_distance

        return predicted_label, posterior, prediction_distance

    def _majority_vote(self, labels, distances):
        best_label = None
        best_count = -1
        best_distance = float("inf")

        for label in sorted(set(labels), key=str):
            label_distances = distances[labels == label]
            count = len(label_distances)
            min_distance = float(np.min(label_distances))
            if count > best_count or (count == best_count and min_distance < best_distance):
                best_label = label
                best_count = count
                best_distance = min_distance

        return best_label

    def _extract_color_and_gray_embeddings(self, face):
        color_face = self._prepare_face(face)
        if color_face is None:
            return []

        gray_face = cv2.cvtColor(color_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

        embeddings = []
        for prepared_face in (color_face, gray_face):
            embedding = self._extract_embedding(prepared_face)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    def _extract_embedding(self, face):
        embedding = np.asarray(self.facenet.predict(face), dtype=np.float32).reshape(-1)
        if embedding.shape[0] != FaceNet.embedding_dimensionality:
            return None
        if not np.all(np.isfinite(embedding)):
            return None

        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm == 0.0:
            return None
        return embedding / norm

    def _prepare_face(self, face):
        if not isinstance(face, np.ndarray) or face.size == 0:
            return None

        if face.ndim == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        elif face.ndim == 3 and face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_BGRA2BGR)
        elif face.ndim != 3 or face.shape[2] != 3:
            return None

        if face.shape[:2] != (224, 224):
            face = cv2.resize(face, dsize=(224, 224))

        return face


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:
    DEFAULT_RANDOM_SEED = 42
    DEFAULT_TOL = 1e-6

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=200):
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)

        # Number of cluster centers for k-means clustering.
        self.num_clusters = max(1, int(num_clusters))
        # Cluster centers.
        self.cluster_center = np.empty(
            (self.num_clusters, FaceNet.embedding_dimensionality), dtype=np.float32
        )
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max(1, int(max_iter))
        self.random_seed = self.DEFAULT_RANDOM_SEED
        self.convergence_tol = self.DEFAULT_TOL
        self.objective_history = []
        self.n_iter_ = 0
        self.converged_ = False
        self.is_fitted_ = False

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )
        self.num_clusters = max(1, int(self.num_clusters))
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32).reshape(
            -1, FaceNet.embedding_dimensionality
        )
        self.cluster_center = np.asarray(self.cluster_center, dtype=np.float32).reshape(
            -1, FaceNet.embedding_dimensionality
        )
        self.cluster_membership = list(self.cluster_membership)
        self.is_fitted_ = (
            len(self.cluster_membership) > 0
            and self.cluster_center.size > 0
            and np.all(np.isfinite(self.cluster_center))
        )

    def partial_fit(self, face):
        embedding = self._extract_embedding(face)
        if embedding is None:
            return None

        if self.embeddings.size == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack((self.embeddings, embedding))
        return None

    def fit(self):
        samples = self._finite_embeddings()
        self.objective_history = []
        self.cluster_membership = []
        self.n_iter_ = 0
        self.converged_ = False

        if len(samples) == 0:
            self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)
            self.cluster_center = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)
            self.is_fitted_ = False
            return None

        self.embeddings = samples
        rng = np.random.default_rng(self.random_seed)
        centers = self._initialize_centers(samples, rng)
        previous_membership = None

        for iteration in range(self.max_iter):
            distances = self._pairwise_distances(samples, centers)
            membership = np.argmin(distances, axis=1)
            membership, centers = self._repair_empty_clusters(samples, membership, centers)

            new_centers = self._update_centers(samples, membership, centers, rng)
            updated_distances = self._pairwise_distances(samples, new_centers)
            updated_membership = np.argmin(updated_distances, axis=1)
            updated_membership, new_centers = self._repair_empty_clusters(
                samples, updated_membership, new_centers
            )
            new_centers = self._update_centers(samples, updated_membership, new_centers, rng)
            updated_distances = self._pairwise_distances(samples, new_centers)
            objective = self._calc_objective(updated_distances, updated_membership)
            self.objective_history.append(objective)

            center_shift = float(np.linalg.norm(new_centers - centers))
            membership_unchanged = (
                previous_membership is not None
                and np.array_equal(updated_membership, previous_membership)
            )

            centers = new_centers
            previous_membership = updated_membership.copy()
            self.n_iter_ = iteration + 1

            if membership_unchanged or center_shift < self.convergence_tol:
                self.converged_ = True
                break

        if previous_membership is None:
            final_distances = self._pairwise_distances(samples, centers)
            final_membership = np.argmin(final_distances, axis=1)
            final_membership, centers = self._repair_empty_clusters(
                samples, final_membership, centers
            )
        else:
            final_membership = previous_membership

        self.cluster_center = centers.astype(np.float32)
        self.cluster_membership = final_membership.astype(int).tolist()
        self.is_fitted_ = True
        self._print_fit_summary()
        return None

    def predict(self, face) -> tuple[int, np.ndarray]:
        embedding = self._extract_embedding(face)
        centers = np.asarray(self.cluster_center, dtype=np.float32).reshape(
            -1, FaceNet.embedding_dimensionality
        )
        valid_centers = np.all(np.isfinite(centers), axis=1)
        if (
            not self.is_fitted_
            or embedding is None
            or centers.size == 0
            or not np.any(valid_centers)
        ):
            return -1, np.empty((0,), dtype=np.float32)

        distances = np.full((len(centers),), np.inf, dtype=np.float32)
        distances[valid_centers] = np.linalg.norm(centers[valid_centers] - embedding, axis=1)
        if not np.any(np.isfinite(distances)):
            return -1, distances
        return int(np.argmin(distances)), distances

    def objective_table(self):
        return [
            (iteration, objective)
            for iteration, objective in enumerate(self.objective_history, start=1)
        ]

    def _extract_embedding(self, face):
        prepared_face = self._prepare_face(face)
        if prepared_face is None:
            return None

        embedding = np.asarray(self.facenet.predict(prepared_face), dtype=np.float32).reshape(-1)
        if embedding.shape[0] != FaceNet.embedding_dimensionality:
            return None
        if not np.all(np.isfinite(embedding)):
            return None

        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm == 0.0:
            return None
        return embedding / norm

    def _prepare_face(self, face):
        if not isinstance(face, np.ndarray) or face.size == 0:
            return None

        if face.ndim == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        elif face.ndim == 3 and face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_BGRA2BGR)
        elif face.ndim != 3 or face.shape[2] != 3:
            return None

        if face.shape[:2] != (224, 224):
            face = cv2.resize(face, dsize=(224, 224))

        return face

    def _finite_embeddings(self):
        embeddings = np.asarray(self.embeddings, dtype=np.float32).reshape(
            -1, FaceNet.embedding_dimensionality
        )
        if embeddings.size == 0:
            return embeddings
        return embeddings[np.all(np.isfinite(embeddings), axis=1)]

    def _initialize_centers(self, samples, rng):
        replace = len(samples) < self.num_clusters
        indices = rng.choice(len(samples), size=self.num_clusters, replace=replace)
        return samples[indices].astype(np.float32).copy()

    def _pairwise_distances(self, samples, centers):
        diff = samples[:, None, :] - centers[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _calc_objective(self, distances, membership):
        assigned_distances = distances[np.arange(len(membership)), membership]
        return float(np.sum(assigned_distances**2))

    def _update_centers(self, samples, membership, centers, rng):
        new_centers = centers.copy()

        for cluster_id in range(self.num_clusters):
            cluster_samples = samples[membership == cluster_id]
            if len(cluster_samples) > 0:
                new_centers[cluster_id] = np.mean(cluster_samples, axis=0)
                continue

            replacement_index = int(rng.integers(len(samples)))
            new_centers[cluster_id] = samples[replacement_index]

        return new_centers.astype(np.float32)

    def _repair_empty_clusters(self, samples, membership, centers):
        membership = np.asarray(membership, dtype=int).copy()
        centers = np.asarray(centers, dtype=np.float32).copy()
        counts = np.bincount(membership, minlength=self.num_clusters)

        for cluster_id in np.flatnonzero(counts == 0):
            donor_mask = counts[membership] > 1
            if not np.any(donor_mask):
                break

            assigned_distances = np.linalg.norm(samples - centers[membership], axis=1)
            assigned_distances[~donor_mask] = -np.inf
            replacement_index = int(np.argmax(assigned_distances))
            old_cluster = int(membership[replacement_index])

            membership[replacement_index] = int(cluster_id)
            centers[cluster_id] = samples[replacement_index]
            counts[old_cluster] -= 1
            counts[cluster_id] += 1

        return membership, centers

    def _print_fit_summary(self):
        print("k-means objective history:")
        for iteration, objective in self.objective_table():
            print("{}\t{:.6f}".format(iteration, objective))
        print("k-means iterations: {}".format(self.n_iter_))
        print("k-means converged: {}".format(self.converged_))
