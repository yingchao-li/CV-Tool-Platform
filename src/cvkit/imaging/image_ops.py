"""Shared image helpers: RAW loading, normalization, tone curves and JPG export."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rawpy


def load_raw_cr3(path):
    """
    Read the visible sensor area of a CR3 file as RAW data.

    Returns a 2-D Bayer array that has not been demosaiced.
    """
    # rawpy parses the camera RAW container; keep only the visible pixels,
    # skipping black borders and calibration regions.
    raw = rawpy.imread(path)
    array = np.array(raw.raw_image_visible).astype(np.float32)
    return array


def save_float_image_as_jpg(img_float, output_path, quality=98):
    """
    Save a float RGB image in [0, 1] as JPG.

    The pipeline stays in float as long as possible and converts to uint8 only
    when exporting.
    """
    # Clip enhanced or normalized values: JPG can only represent 0-255.
    img_float = np.clip(img_float, 0.0, 1.0)
    img_uint8 = (img_float * 255.0).astype(np.uint8)
    Image.fromarray(img_uint8).save(output_path, quality=quality)


def show_image(img, title):
    """Display a float RGB image, mainly for debugging and inspecting results."""
    plt.figure(figsize=(10, 6))
    plt.imshow(np.clip(img, 0.0, 1.0))
    plt.title(title)
    plt.axis("off")
    plt.show()


def percentile_normalize(data, low=0.01, high=99.99):
    """
    Normalize data to [0, 1] using percentiles.

    This ignores a few extreme dark or bright outliers and is more stable than
    plain min/max.
    """
    # a and b are the black and white reference points.
    a = np.percentile(data, low)
    b = np.percentile(data, high)

    # Avoid division by zero when the image has almost no dynamic range.
    if abs(b - a) < 1e-8:
        normalized = np.zeros_like(data, dtype=np.float32)
    else:
        normalized = (data - a) / (b - a)

    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32), a, b


def gamma_correction(data, gamma=0.3):
    """
    Apply gamma correction to a normalized image:
        y = x^gamma

    A gamma below 1 brightens the shadows.
    """
    data = np.clip(data, 0.0, 1.0)
    return np.power(data, gamma).astype(np.float32)


def apply_brightness_contrast(rgb, brightness=0.0, contrast=1.0):
    """
    Final display adjustment on a [0, 1] float image: brightness is an additive
    offset and contrast scales around 0.5.
    """
    img = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    img = img + brightness
    img = np.clip(img, 0.0, 1.0)
    img = (img - 0.5) * contrast + 0.5
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def log_curve(data, alpha=10.0):
    """
    Non-gamma tone curve:
        y = log(1 + alpha*x) / log(1 + alpha)

    The log curve compresses highlights while keeping shadows visible.
    """
    data = np.clip(data, 0.0, 1.0)
    return (np.log1p(alpha * data) / np.log1p(alpha)).astype(np.float32)

def sigmoid_curve(data, k=8.0):
    data = np.clip(data, 0.0, 1.0)
    y = 1.0 / (1.0 + np.exp(-k * (data - 0.5)))

    # Renormalize so that 0 stays 0 and 1 stays 1.
    y0 = 1.0 / (1.0 + np.exp(-k * (0.0 - 0.5)))
    y1 = 1.0 / (1.0 + np.exp(-k * (1.0 - 0.5)))
    y = (y - y0) / (y1 - y0)

    return y.astype(np.float32)