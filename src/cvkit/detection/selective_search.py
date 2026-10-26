"""Selective Search region proposals.

Felzenszwalb over-segmentation followed by greedy merging using colour,
texture, size and fill similarity.  The implementation follows the original
Selective Search formulation and returns every region seen during merging,
so callers can filter proposals by area or aspect ratio.
"""

from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    """Over-segment the image with Felzenszwalb's algorithm.

    The integer region labels are appended as a fourth channel so that the
    merging stage can slice every region by label.
    """
    # 1.1. Generate the initial image mask.
    # Use skimage.segmentation.felzenszwalb for initial segmentation; it returns
    # the region label of each pixel.
    seg_mask = skimage.segmentation.felzenszwalb(
        im_orig,
        scale=scale,
        sigma=sigma,
        min_size=min_size
    )

    # 1.2. Merge the image mask into the image as a 4th channel.
    # Expand the 2D mask (H, W) to 3D (H, W, 1) so it can be concatenated with
    # the original image along the channel axis.
    seg_mask_expanded = np.expand_dims(seg_mask, axis=-1)

    # Concatenate the original image (H, W, 3) and the expanded mask
    # (H, W, 1) along the last dimension to get (H, W, 4).
    im_new = np.concatenate((im_orig, seg_mask_expanded), axis=-1)

    return im_new

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    # Sum the per-bin minimum to measure how much the two color distributions
    # overlap.
    return np.sum(np.minimum(r1["hist_c"], r2["hist_c"]))


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    # Like color similarity, measure how much the texture distributions overlap.
    return np.sum(np.minimum(r1["hist_t"], r2["hist_t"]))


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    # Size similarity favors merging small regions early, which prevents a
    # single region from swallowing the whole image.
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    # Fill similarity measures how tightly one bounding box can enclose both
    # regions; a tighter box means higher similarity.
    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]) + 1) *
            (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]) + 1)
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """Colour histogram of one region.

    One 25-bin histogram per HSV channel, L1-normalised, matching the bin
    count used in the original Selective Search paper.
    """
    BINS = 25
    hist = np.array([])
    # Iterate over the 3 color channels to compute histograms.
    for colour_channel in (0, 1, 2):
        c = img[:, colour_channel]
        # HSV values in skimage are floats in [0.0, 1.0].
        hist_channel, _ = np.histogram(c, bins=BINS, range=(0.0, 1.0))
        hist = np.concatenate([hist, hist_channel])

    # L1 normalization.
    hist_sum = np.sum(hist)
    if hist_sum != 0:
        hist = hist / hist_sum

    return hist

def calc_texture_gradient(img):
    """LBP texture map for the whole image.

    Selective Search uses Gaussian derivatives at eight orientations; this
    implementation uses an 8-neighbour, radius-1 local binary pattern, which
    is cheaper and adequate for the coarse similarity signal.
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    # Use LBP (Local Binary Pattern) to extract texture features from each
    # channel independently.
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], P=8, R=1.0
        )
    return ret

def calc_texture_hist(img):
    """Texture histogram of one region (10 bins per channel, L1-normalised)."""
    BINS = 10
    hist = np.array([])
    # The output of LBP with P=8 usually falls in [0, 255].
    for colour_channel in (0, 1, 2):
        c = img[:, colour_channel]
        hist_channel, _ = np.histogram(c, bins=BINS, range=(0, 256))
        hist = np.concatenate([hist, hist_channel])

    # L1 normalization.
    hist_sum = np.sum(hist)
    if hist_sum != 0:
        hist = hist / hist_sum

    return hist

def extract_regions(img):
    """Build the region table ``R`` used by the merging stage.

    Each entry stores the HSV pixels, pixel mask, size, bounding box and the
    colour/texture histograms. The texture gradient is computed once for the
    whole image and indexed per region.
    """
    R = {}
    # 1. Extract the image and convert it to HSV color space.
    # The first 3 channels are image pixels; the 4th channel is the region
    # label.
    rgb_img = img[:, :, :3]
    hsv_img = skimage.color.rgb2hsv(rgb_img)

    # 2. Compute the texture gradient for the whole image.
    tex_grad = calc_texture_gradient(rgb_img)

    # 3. Extract all labels and iterate over the unique regions.
    labels = img[:, :, 3].astype(int)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Use a numpy mask to quickly get the coordinates of the pixels that
        # belong to the current label.
        mask = (labels == label)
        y_coords, x_coords = np.where(mask)

        # Extract the HSV and texture pixels of this region for the histograms.
        hsv_pixels = hsv_img[mask]
        tex_pixels = tex_grad[mask]

        # Build and store the region property dictionary.
        R[label] = {
            "min_x": np.min(x_coords), "max_x": np.max(x_coords),
            "min_y": np.min(y_coords), "max_y": np.max(y_coords),
            "size": len(x_coords),
            "hist_c": calc_colour_hist(hsv_pixels),
            "hist_t": calc_texture_hist(tex_pixels),
            "labels": [label]
        }
    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    # Extract all (label, region_data) pairs from the dictionary into a list.
    region_items = list(regions.items())

    # Iterate over all possible region pairs.
    # Only compare i and j with j > i to avoid duplicate checks and
    # self-comparison.
    for i in range(len(region_items)):
        for j in range(i + 1, len(region_items)):
            ai, a = region_items[i]
            bi, b = region_items[j]

            # Check whether the two bounding boxes intersect using the provided
            # intersect function.
            if intersect(a, b):
                # If they intersect, append the pair to the neighbours list in
                # the required format.
                neighbours.append(((ai, a), (bi, b)))

    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    # 1. Compute the bounding box of the merged region.
    # Take the minimum of the min coordinates and the maximum of the max
    # coordinates as the new boundary.
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])

    # 2. Store the pixel size of the new region.
    rt["size"] = new_size

    # 3. Merge the color and texture histograms with a size-weighted average.
    # As in the Selective Search paper, the new histograms are the original
    # histograms weighted by their region sizes.
    rt["hist_c"] = (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size
    rt["hist_t"] = (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size

    # 4. Concatenate the label lists of the original regions.
    # Track which initial over-segmented regions make up this composite region.
    rt["labels"] = r1["labels"] + r2["labels"]
    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Over-segment the image and append the label map as a fourth channel.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Build the region descriptors (colour, texture, size, fill).
    R = extract_regions(image)

    # Build the region adjacency graph.
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Merge the most similar pair into a new region.
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Drop every similarity entry that refers to i or j.
        # Collect every similarity key that references the merged regions i or j.
        keys_to_delete = []
        for key in S.keys():
            # A key is a (region_a, region_b) tuple.
            if i in key or j in key:
                keys_to_delete.append(key)

        # Remove the stale entries from the similarity table.
        # Remove all keys that reference i or j from the similarity dictionary.
        for key in keys_to_delete:
            del S[key]

        # Link the merged region to its new neighbours.
        # Extract all potential neighbors of the new region t.
        # A set removes duplicates automatically.
        new_neighbors = set()

        for key in keys_to_delete:
            # Each key is a stale (region_a, region_b) pair removed above.
            # A region in the pair that is neither i nor j becomes a new
            # neighbor of t.
            if key[0] != i and key[0] != j:
                new_neighbors.add(key[0])
            if key[1] != i and key[1] != j:
                new_neighbors.add(key[1])

        # For every new neighbor, compute the similarity with the new region t.
        for n in new_neighbors:
            # Compute the similarity and add it to the dictionary S.
            S[(t, n)] = calc_sim(R[t], R[n], imsize)


    # Export every region seen during merging as a proposal.
    regions = []
    # Iterate over every candidate region produced by the merging.
    for k, r in R.items():
        # Convert the min/max coordinates to (x, y, width, height) format for
        # matplotlib drawing.
        width = r["max_x"] - r["min_x"]
        height = r["max_y"] - r["min_y"]

        # Package the data as required by the selective_search return value.
        regions.append({
            'rect': (r["min_x"], r["min_y"], width, height),
            'size': r["size"],
            'labels': r["labels"]
        })

    return image, regions


