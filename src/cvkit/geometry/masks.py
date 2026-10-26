"""Utility functions for cleaning and manipulating binary image masks."""

import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy import ndimage

def create_mask_from_indices(image_shape, rows, cols, selected_mask):
    """Map a 1D inlier mask back onto a 2D image mask.

    Parameters:
        image_shape: image size, usually PC.shape[:2]
        rows, cols: image coordinates of the valid points
        selected_mask: boolean inlier mask returned by RANSAC

    Returns:
        A 2D boolean mask where True marks an inlier.
    """

    if len(rows) != len(cols):
        raise ValueError("rows and cols must have the same length.")

    if len(selected_mask) != len(rows):
        raise ValueError(
            f"selected_mask length {len(selected_mask)} does not match "
            f"rows/cols length {len(rows)}."
        )

    mask = np.zeros(image_shape, dtype=bool)
    mask[rows[selected_mask], cols[selected_mask]] = True

    return mask


def remove_small_true_components(mask, min_size=100):
    """Remove True connected components smaller than min_size.

    This step discards isolated noise blobs.
    """

    structure = generate_binary_structure(2, 2)
    labeled_mask, num_components = label(mask, structure=structure)

    if num_components == 0:
        return np.zeros_like(mask, dtype=bool)

    cleaned = np.zeros_like(mask, dtype=bool)

    for component_id in range(1, num_components + 1):
        component = labeled_mask == component_id
        size = np.sum(component)

        if size >= min_size:
            cleaned[component] = True

    return cleaned


def fill_small_false_holes(mask, max_hole_size=300):
    """Fill only the small False holes enclosed inside a mask.

    Large holes are left untouched, so box regions (comparatively large False
    areas) are preserved.

    Parameters:
        mask: 2D boolean mask, True=floor, False=non-floor
        max_hole_size: only fill holes whose area is at most this threshold and
            that do not touch the image border

    Returns:
        The mask with small holes filled.
    """

    false_mask = ~mask

    structure = generate_binary_structure(2, 2)
    labeled_false, num_components = label(false_mask, structure=structure)

    result = mask.copy()

    for component_id in range(1, num_components + 1):
        component = labeled_false == component_id
        size = np.sum(component)

        # If a false region touches the image border, it is probably background,
        # not an internal hole.
        touches_border = (
            np.any(component[0, :]) or
            np.any(component[-1, :]) or
            np.any(component[:, 0]) or
            np.any(component[:, -1])
        )

        # Fill only small internal holes.
        if (not touches_border) and size <= max_hole_size:
            result[component] = True

    return result


def filter_floor_mask(mask, min_floor_size=100, max_hole_size=300):
    """Clean a raw floor mask, reducing noise while keeping box regions.

    Input semantics:
        True=floor, False=non-floor

    Steps:
        1. Remove small floor connected components;
        2. Fill small internal holes.
    """

    cleaned = mask.astype(bool)

    # Build the structuring element used for the morphological pass.
    kernel = ndimage.generate_binary_structure(2, 2)
    kernel_size = 5

    # Closing first, opening after.
    dilated = ndimage.binary_dilation(cleaned, structure=kernel, iterations=kernel_size)
    closed = ndimage.binary_erosion(dilated, structure=kernel, iterations=kernel_size)
    eroded = ndimage.binary_erosion(closed, structure=kernel, iterations=kernel_size)
    cleaned = ndimage.binary_dilation(eroded, structure=kernel, iterations=kernel_size)

    cleaned = remove_small_true_components(
        cleaned,
        min_size=min_floor_size
    )

    cleaned = fill_small_false_holes(
        cleaned,
        max_hole_size=max_hole_size
    )

    return cleaned.astype(bool)


def get_non_floor_mask(floor_mask):
    """Derive the non-floor mask from a floor mask.

    Input:
        floor_mask: True=floor, False=non-floor

    Output:
        non_floor_mask: True=non-floor, False=floor
    """

    return ~floor_mask.astype(bool)


def get_points_from_mask(PC, mask):
    """Extract 3D points from a point cloud using a 2D mask.

    Parameters:
        PC: H x W x 3 point cloud
        mask: H x W boolean mask

    Returns:
        points: N x 3 3D points
        rows, cols: image coordinates of the selected points
    """

    if PC.shape[:2] != mask.shape:
        raise ValueError(
            f"PC shape {PC.shape[:2]} and mask shape {mask.shape} do not match."
        )

    rows, cols = np.where(mask)
    points = PC[rows, cols, :]

    # Remove invalid measurements.
    valid = points[:, 2] != 0

    return points[valid], rows[valid], cols[valid]


def largest_connected_component(mask):
    """Keep only the largest True connected component.

    Used to extract the main box-top region. Adjacent pixels are grouped into
    distinct labeled clusters, and the cluster with the most pixels is returned.
    """

    structure = generate_binary_structure(2, 2)
    labeled_mask, num_components = label(mask, structure=structure)

    if num_components == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(labeled_mask.ravel())

    # Label 0 is background, ignore it.
    component_sizes[0] = 0

    largest_label = np.argmax(component_sizes)

    return labeled_mask == largest_label


def remove_small_components(mask, min_size=100):
    """Generic alias: keep True components of at least min_size pixels."""

    return remove_small_true_components(mask, min_size=min_size)