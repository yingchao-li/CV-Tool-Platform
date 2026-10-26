from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class FaceDetectionResult:
    image: np.ndarray
    """The image."""
    rect: tuple[int, int, int, int]
    """The face bounding box (top left x, top left y, width, height)."""
    aligned: np.ndarray
    """The aligned face image."""


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(
        self, tm_window_size: int = 100, tm_threshold: float = 0.0, aligned_image_size: int = 224
    ) -> None:
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference: Optional[FaceDetectionResult] = None
        self.template: Optional[np.ndarray] = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # Template matching parameters.
        self.tm_method = cv2.TM_CCOEFF_NORMED
        self.tm_threshold = tm_threshold if tm_threshold > 0.0 else 0.55
        self.tm_search_margin = max(1, min(25, tm_window_size // 4))

    def track_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        if not self._valid_image(image):
            self._clear_reference()
            return None

        if self.reference is None or self.template is None:
            return self.detect_face(image)

        reference_rect = self._clip_rect(image, self.reference.rect)
        if reference_rect is None or not self._valid_template(self.template):
            return self.detect_face(image)

        search_rect = self._search_rect(image, reference_rect)
        if search_rect is None:
            return self.detect_face(image)

        search_left, search_top, search_width, search_height = search_rect
        search_region = image[
            search_top : search_top + search_height,
            search_left : search_left + search_width,
        ]
        template_height, template_width = self.template.shape[:2]
        if search_height < template_height or search_width < template_width:
            return self.detect_face(image)

        search_gray = self._to_gray(search_region)
        result = cv2.matchTemplate(search_gray, self.template, self.tm_method)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        if not np.isfinite(max_value) or max_value < self.tm_threshold:
            return self.detect_face(image)

        face_rect = self._clip_rect(
            image,
            (
                search_left + max_location[0],
                search_top + max_location[1],
                template_width,
                template_height,
            ),
        )
        if face_rect is None:
            return self.detect_face(image)

        face = self._build_result(image, face_rect)
        if face is None:
            return self.detect_face(image)

        self._store_reference(face)
        return face

    # Face detection in a new image.
    def detect_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        if not self._valid_image(image):
            self._clear_reference()
            return None

        # Retrieve all detectable faces in the given image.
        if not (
            detections := self.detector.detect_faces(image, threshold_pnet=0.85, threshold_rnet=0.9)
        ):
            self._clear_reference()
            return None

        # Select face with the largest bounding box.
        faces = [
            face
            for detection in detections
            if (face := self._build_result(image, detection["box"])) is not None
        ]
        if not faces:
            self._clear_reference()
            return None

        face = max(faces, key=lambda result: result.rect[2] * result.rect[3])
        self._store_reference(face)
        return face

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        face = self.crop_face(image, face_rect)
        if face.size == 0:
            return np.empty((0, 0, 3), dtype=image.dtype)
        return cv2.resize(face, dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        clipped_rect = self._clip_rect(image, face_rect)
        if clipped_rect is None:
            return np.empty((0, 0, 3), dtype=image.dtype)
        left, top, width, height = clipped_rect
        bottom = top + height
        right = left + width
        if image.ndim == 2:
            return image[top:bottom, left:right]
        return image[top:bottom, left:right, :]

    def _valid_image(self, image: np.ndarray) -> bool:
        return (
            isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim in (2, 3)
            and image.shape[0] > 1
            and image.shape[1] > 1
        )

    def _clip_rect(self, image: np.ndarray, face_rect) -> Optional[tuple[int, int, int, int]]:
        if not self._valid_image(image) or face_rect is None or len(face_rect) != 4:
            return None

        x, y, width, height = (int(round(value)) for value in face_rect)
        if width <= 1 or height <= 1:
            return None

        image_height, image_width = image.shape[:2]
        left = max(x, 0)
        top = max(y, 0)
        right = min(x + width, image_width)
        bottom = min(y + height, image_height)
        clipped_width = right - left
        clipped_height = bottom - top
        if clipped_width <= 1 or clipped_height <= 1:
            return None

        return left, top, clipped_width, clipped_height

    def _search_rect(
        self, image: np.ndarray, face_rect: tuple[int, int, int, int]
    ) -> Optional[tuple[int, int, int, int]]:
        left, top, width, height = face_rect
        image_height, image_width = image.shape[:2]

        search_left = max(left - self.tm_search_margin, 0)
        search_top = max(top - self.tm_search_margin, 0)
        search_right = min(left + width + self.tm_search_margin, image_width)
        search_bottom = min(top + height + self.tm_search_margin, image_height)
        search_width = search_right - search_left
        search_height = search_bottom - search_top
        if search_width <= 1 or search_height <= 1:
            return None

        return search_left, search_top, search_width, search_height

    def _build_result(
        self, image: np.ndarray, face_rect
    ) -> Optional[FaceDetectionResult]:
        clipped_rect = self._clip_rect(image, face_rect)
        if clipped_rect is None:
            return None

        face = self.crop_face(image, clipped_rect)
        if face.size == 0:
            return None

        aligned = self.align_face(image, clipped_rect)
        if aligned.size == 0:
            return None

        return FaceDetectionResult(image=image, rect=clipped_rect, aligned=aligned)

    def _store_reference(self, face: FaceDetectionResult) -> None:
        template = self.crop_face(face.image, face.rect)
        if template.size == 0:
            self._clear_reference()
            return

        self.template = self._to_gray(template).copy()
        self.reference = FaceDetectionResult(
            image=face.image.copy(),
            rect=face.rect,
            aligned=face.aligned.copy(),
        )

    def _clear_reference(self) -> None:
        self.reference = None
        self.template = None

    def _valid_template(self, template: np.ndarray) -> bool:
        return (
            isinstance(template, np.ndarray)
            and template.size > 0
            and template.shape[0] > 1
            and template.shape[1] > 1
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
