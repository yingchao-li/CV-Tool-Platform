
"""Plane-based estimation of box height, length, width, and corners."""

import numpy as np


def plane_distance(plane1, plane2):
    """Compute the distance between two nearly parallel planes.

    Plane format:
        normal · x = d
        plane = (normal, d)

    Note:
        RANSAC already normalizes the normals, but they are normalized again
        here for safety.
    """

    n1, d1 = plane1
    n2, d2 = plane2

    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)

    # Normalize again for safety
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # If normals point in opposite directions, flip the second plane
    if np.dot(n1, n2) < 0:
        n2 = -n2
        d2 = -d2

    height = abs(d1 - d2)

    return height


def get_box_top_points(PC, box_top_mask):
    """Extract the valid 3D points covered by the box-top mask.

    Parameters:
        PC: H x W x 3 point cloud
        box_top_mask: H x W boolean mask (True = box top)

    Returns:
        points: N x 3
    """

    if PC.shape[:2] != box_top_mask.shape:
        raise ValueError(
            f"PC shape {PC.shape[:2]} and mask shape {box_top_mask.shape} do not match."
        )

    points = PC[box_top_mask]

    # Remove invalid or non-finite measurements.
    valid = (points[:, 2] != 0) & np.all(np.isfinite(points), axis=1)
    points = points[valid]

    if points.shape[0] == 0:
        raise ValueError("No valid box top points found.")

    return points


def estimate_length_width_simple(PC, box_top_mask):
    """Compute length and width directly from the x and y extents.

    Disadvantage:
        If the box is rotated, the result may be overestimated.
    """

    points = get_box_top_points(PC, box_top_mask)

    xs = points[:, 0]
    ys = points[:, 1]

    length = np.max(xs) - np.min(xs)
    width = np.max(ys) - np.min(ys)

    # Put larger one as length, smaller one as width
    length, width = sorted([length, width], reverse=True)

    return length, width

def estimate_length_width_pca(PC, box_top_mask):
    """Estimate length and width from the principal directions of the box top.

    1. Use PCA to get the two dominant directions of the box-top points; one is
       the width direction and the other is the length direction.
    2. Project the points onto these two dominant directions.
    3. Use the maximum spread along each direction as the length and width.

    Advantage:
        Estimate the true dimensions of an object regardless of its orientation
        by aligning the measurement axes with the object's principal directions
        of spread rather than the fixed image coordinates.
    """

    points = get_box_top_points(PC, box_top_mask)

    # Center points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD:
    # Vt[0] = first principal direction
    # Vt[1] = second principal direction
    # Vt[2] = normal-like direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    dir1 = vt[0]
    dir2 = vt[1]

    # Project points onto the two dominant directions
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        proj1 = centered @ dir1
        proj2 = centered @ dir2

    valid_projection = np.isfinite(proj1) & np.isfinite(proj2)
    if not np.any(valid_projection):
        raise ValueError("No finite PCA projections found for box top points.")

    proj1 = proj1[valid_projection]
    proj2 = proj2[valid_projection]

    size1 = np.max(proj1) - np.min(proj1)
    size2 = np.max(proj2) - np.min(proj2)

    length, width = sorted([size1, size2], reverse=True)

    return length, width

def estimate_box_dimensions(
    PC,
    floor_model,
    box_model,
    box_top_mask,
    method="pca",
):
    """Estimate the height, length, and width of the box.

    Parameters:
        PC: H x W x 3 point cloud
        floor_model: (normal, d)
        box_model: (normal, d)
        box_top_mask: H x W boolean mask of the box top
        method: "pca" or "simple"

    Returns:
        height, length, width
    """

    height = plane_distance(floor_model, box_model)

    if method == "pca":
        length, width = estimate_length_width_pca(PC, box_top_mask)
    elif method == "simple":
        length, width = estimate_length_width_simple(PC, box_top_mask)
    return height, length, width


def get_box_top_corners_pca(PC, box_top_mask):
    """Estimate the four 3D corners of the box top from PCA coordinates.

    Returns:
        corners: 4 x 3 approximate corner coordinates

    Note:
        The method builds a bounding rectangle in PCA space and maps it back to
        3D, so the result is approximate.
    """

    points = get_box_top_points(PC, box_top_mask)

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    dir1 = vt[0] # long-edge direction
    dir2 = vt[1] # short-edge direction

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        proj1 = centered @ dir1
        proj2 = centered @ dir2

    valid_projection = np.isfinite(proj1) & np.isfinite(proj2)
    if not np.any(valid_projection):
        raise ValueError("No finite PCA projections found for box top corners.")

    proj1 = proj1[valid_projection]
    proj2 = proj2[valid_projection]

    min1, max1 = np.min(proj1), np.max(proj1)
    min2, max2 = np.min(proj2), np.max(proj2)

    corners = np.array([
        centroid + min1 * dir1 + min2 * dir2,
        centroid + max1 * dir1 + min2 * dir2,
        centroid + max1 * dir1 + max2 * dir2,
        centroid + min1 * dir1 + max2 * dir2,
    ])

    return corners
