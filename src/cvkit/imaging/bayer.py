"""Simple Bayer demosaicing using channel masks and convolution."""

import numpy as np
from scipy.ndimage import convolve


def create_bayer_masks(shape, pattern="RGGB"):
    """
    Create R/G/B sampling masks for a Bayer pattern.

    The characters of ``pattern`` describe the 2x2 Bayer tile:

        pattern[0] pattern[1]
        pattern[2] pattern[3]

    Example:
        "RGGB" means:

            R G
            G B

        "GBRG" means:

            G B
            R G
    """
    h, w = shape

    # A value of 1 marks a sampled pixel; 0 marks a pixel to interpolate.
    r_mask = np.zeros((h, w), dtype=np.float32)
    g_mask = np.zeros((h, w), dtype=np.float32)
    b_mask = np.zeros((h, w), dtype=np.float32)

    # The four start offsets are the four pixels of the 2x2 Bayer tile.
    positions = [
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    ]

    for row_start, col_start, color in positions:
        if color == "R":
            r_mask[row_start::2, col_start::2] = 1.0
        elif color == "G":
            g_mask[row_start::2, col_start::2] = 1.0
        elif color == "B":
            b_mask[row_start::2, col_start::2] = 1.0
        else:
            raise ValueError(f"Unknown color in Bayer pattern: {color}")

    return r_mask, g_mask, b_mask


def interpolate_channel(raw, mask, kernel):
    """
    Bilinear interpolation of a single channel:

        C = ((Mc * X) convolved with K) / (Mc convolved with K)

    ``raw`` is the Bayer image and ``mask`` marks the samples of one channel.
    """
    # Numerator: gather real samples of this channel in the neighbourhood.
    numerator = convolve(raw * mask, kernel, mode="mirror")

    # Denominator: count those samples so the result is their average.
    denominator = convolve(mask, kernel, mode="mirror")
    channel = numerator / np.maximum(denominator, 1e-8)
    return channel.astype(np.float32)


def demosaic_simple(raw, pattern="RGGB", kernel_size=3):
    """
    Simple demosaicing by masked convolution and averaging.

    This is not a high-quality commercial algorithm, but it demonstrates the
    basic Bayer-to-RGB process.
    """
    r_mask, g_mask, b_mask = create_bayer_masks(raw.shape, pattern)

    # An all-ones kernel averages over the local window.
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    # Interpolate each channel to full resolution.
    r = interpolate_channel(raw, r_mask, kernel)
    g = interpolate_channel(raw, g_mask, kernel)
    b = interpolate_channel(raw, b_mask, kernel)

    rgb = np.stack([r, g, b], axis=2).astype(np.float32)
    return rgb
