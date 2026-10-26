"""Visualization helpers for amplitude, distance, mask, and detection results."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2


def normalize_display_range(image, lower_percentile=1, upper_percentile=99):
    """Compute a robust display range for image visualization.

    Percentile clipping suppresses extreme values so the image does not end up
    with too little contrast.
    """
    valid = image[np.isfinite(image)]

    if valid.size == 0:
        return None, None

    vmin = np.percentile(valid, lower_percentile)
    vmax = np.percentile(valid, upper_percentile)

    return vmin, vmax


def show_amplitude_image(A):
    """Show the amplitude image with a percentile-based contrast stretch."""
    vmin, vmax = normalize_display_range(A, 1, 99)

    plt.figure()
    plt.imshow(A, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Amplitude Image")
    plt.colorbar()
    plt.show()


def show_distance_image(D):
    """Show the distance image with a percentile-based contrast stretch."""
    # In a distance image, 0 usually marks invalid pixels, so keep it out of the
    # display range computation.
    D_display = D.copy()
    D_display[D_display == 0] = np.nan

    vmin, vmax = normalize_display_range(D_display, 1, 99)

    plt.figure()
    plt.imshow(D_display, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Distance Image")
    plt.colorbar()
    plt.show()


def show_images(A, D):
    """Show the amplitude and distance images side by side."""

    A_display = A.copy()

    D_display = D.copy()
    D_display[D_display == 0] = np.nan

    a_vmin, a_vmax = normalize_display_range(A_display, 1, 99)
    d_vmin, d_vmax = normalize_display_range(D_display, 1, 99)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(A_display, cmap="gray", vmin=a_vmin, vmax=a_vmax)
    plt.title("Amplitude Image")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(D_display, cmap="gray", vmin=d_vmin, vmax=d_vmax)
    plt.title("Distance Image")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def show_point_cloud(PC, step=5):
    """Subsample the point cloud and show it as a 3D scatter plot."""

    pc = PC[::step, ::step, :]

    x = pc[:, :, 0].reshape(-1)
    y = pc[:, :, 1].reshape(-1)
    z = pc[:, :, 2].reshape(-1)

    valid = z != 0

    x = x[valid]
    y = y[valid]
    z = z[valid]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, s=1)

    ax.set_title("Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def show_mask(mask, title="Mask"):
    """Display a binary mask."""
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.show()

def show_final_result_3d(PC, floor_mask, box_top_mask, corners=None, step=4):
    """Show floor points, box-top points, and corners in 3D."""

    import numpy as np
    import matplotlib.pyplot as plt

    # subsample for speed
    PC_sub = PC[::step, ::step, :]
    floor_sub = floor_mask[::step, ::step]
    box_sub = box_top_mask[::step, ::step]

    valid = PC_sub[:, :, 2] != 0

    floor_points = PC_sub[floor_sub & valid]
    box_points = PC_sub[box_sub & valid]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if floor_points.shape[0] > 0:
        ax.scatter(
            floor_points[:, 0],
            floor_points[:, 1],
            floor_points[:, 2],
            s=1,
            label="Floor"
        )

    if box_points.shape[0] > 0:
        ax.scatter(
            box_points[:, 0],
            box_points[:, 1],
            box_points[:, 2],
            s=5,
            label="Box Top"
        )

    if corners is not None:
        corners = np.asarray(corners)

        ax.scatter(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            s=50,
            marker="o",
            label="Corners"
        )

        # connect corners
        closed = np.vstack([corners, corners[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            closed[:, 2],
            linewidth=2,
            label="Box Top Boundary"
        )

    ax.set_title("Final Detection Result")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()
def get_rectangle_from_mask(mask):
    """Estimate the box-top rectangle as four image-space corners.

    Parameters:
        mask: H x W boolean mask where True marks the box-top region

    Returns:
        corners: 4 x 2 array, each row is [col, row], ready for plotting
    """
    # 1. Using ndimage.label to get the largest connected component
    labeled_mask, num_features = label(mask)
    region_counts = np.bincount(labeled_mask.ravel())
    main_label = np.argmax(region_counts[1:]) + 1

    # Extract the main mask
    main_mask = (labeled_mask == main_label).astype(np.uint8)
    # 2. Looking for Contours
    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    cnt = max(contours, key=cv2.contourArea)

    # 1. Calculate contour perimeter to establish a baseline for approximation precision
    # 'True' indicates the contour is a closed loop.
    peri = cv2.arcLength(cnt, True)

    # 2. Perform polygon approximation
    # 'eps' (epsilon) is the maximum distance between the original curve and its approximation.
    eps = 0.02 * peri
    approx = cv2.approxPolyDP(cnt, eps, True)

    # Reshape (N, 1, 2) to (N, 2)
    approx = approx.squeeze()

    # 3. Handle cases where the approximation produces more than 4 vertices (complex shapes)
    if len(approx) > 4:
        # Use a Convex Hull to eliminate concave noise and simplify the external shape
        hull = cv2.convexHull(cnt).squeeze()
        # Re-run approximation on the hull with a stricter tolerance (4%) to force a 4-corner result
        approx = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True).squeeze()

    # 4. If the approximation has fewer than 4 vertices, reuse minAreaRect()
    # to find the minimum-area rectangle.
    if len(approx) != 4:
        rect = cv2.minAreaRect(cnt)
        corners = cv2.boxPoints(rect)
    else:
        corners = approx.astype(np.float64)
    return corners
def show_final_result_2d(floor_mask, box_top_mask, title="Final Detection Result"):
    """Render the final 2D result visualization.

    Colors:
        green = floor
        blue = non-floor / background / box sides
        red = detected box top
        cyan = estimated box-top boundary
    """

    if floor_mask.shape != box_top_mask.shape:
        raise ValueError(
            f"floor_mask shape {floor_mask.shape} and box_top_mask shape "
            f"{box_top_mask.shape} do not match."
        )

    h, w = floor_mask.shape

    # RGB image
    result = np.zeros((h, w, 3), dtype=float)

    # Colors
    floor_color = np.array([0.55, 1.00, 0.40])      # light green
    non_floor_color = np.array([0.00, 0.00, 0.45])  # dark blue
    box_top_color = np.array([0.60, 0.00, 0.00])    # dark red

    # Default: non-floor / background
    result[:, :] = non_floor_color

    # Floor
    result[floor_mask] = floor_color

    # Box top overwrites floor/non-floor
    result[box_top_mask] = box_top_color

    # Estimate rectangle from box top mask
    corners = get_rectangle_from_mask(box_top_mask)

    # Close polygon
    closed_corners = np.vstack([corners, corners[0]])

    plt.figure(figsize=(10, 7))
    plt.imshow(result)
    plt.title(title, fontsize=18)

    # Draw cyan boundary
    plt.plot(
        closed_corners[:, 0],
        closed_corners[:, 1],
        color="cyan",
        linewidth=3,
    )

    # Draw corner points
    plt.scatter(
        corners[:, 0],
        corners[:, 1],
        color="cyan",
        s=30,
    )

    # Add labels around the box
    center = np.mean(corners, axis=0)

    # Use image-coordinate rectangle sides
    top_point = corners[np.argmin(corners[:, 1])]
    bottom_point = corners[np.argmax(corners[:, 1])]
    left_point = corners[np.argmin(corners[:, 0])]
    right_point = corners[np.argmax(corners[:, 0])]

    plt.text(
        top_point[0],
        top_point[1] - 35,
        "top",
        color="black",
        fontsize=12,
        ha="center",
    )

    plt.text(
        bottom_point[0],
        bottom_point[1] + 35,
        "bottom",
        color="black",
        fontsize=12,
        ha="center",
    )

    plt.text(
        left_point[0] - 45,
        left_point[1],
        "left",
        color="black",
        fontsize=12,
        ha="center",
    )

    plt.text(
        right_point[0] + 45,
        right_point[1],
        "right",
        color="black",
        fontsize=12,
        ha="center",
    )

    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.axis("off")
    plt.tight_layout()
    plt.show()