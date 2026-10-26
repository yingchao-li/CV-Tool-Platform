"""Estimate the Bayer phase from a numpy RAW and its camera JPG.

The pipeline has four steps:
1. Read the RAW array and the camera JPG.
2. Map JPG coordinates onto RAW coordinates.
3. Detect the red, green and blue pen regions in the JPG.
4. Score RGGB/GRBG/GBRG/BGGR by their RAW response in those regions.

The highest-scoring pattern is taken as the estimated Bayer phase.
"""
import numpy as np
import matplotlib.pyplot as plt

from .config import SAMPLE_NPY, SAMPLE_JPG


def load_data(raw_path, jpg_path):
    """Load the numpy RAW and its camera JPEG reference."""
    raw = np.load(raw_path).astype(np.float32)
    jpg = plt.imread(jpg_path).astype(np.float32)

    # matplotlib may return [0, 1] for a JPG; scale to [0, 255] for thresholding.
    if jpg.max() <= 1.0:
        jpg *= 255.0

    return raw, jpg


def map_jpg_to_raw_coordinates(raw, jpg):
    """
    Coarsely map the JPG image to RAW coordinates.

    RAW and JPG resolutions are close but not identical, so proportional
    scaling gives an approximate match.
    """
    raw_h, raw_w = raw.shape
    jpg_h, jpg_w = jpg.shape[:2]

    # yy/xx are RAW pixel coordinates; yy_jpg/xx_jpg are the matching JPG samples.
    yy, xx = np.indices(raw.shape)

    yy_jpg = np.clip((yy * jpg_h / raw_h).astype(int), 0, jpg_h - 1)
    xx_jpg = np.clip((xx * jpg_w / raw_w).astype(int), 0, jpg_w - 1)

    jpg_on_raw = jpg[yy_jpg, xx_jpg]

    return jpg_on_raw, yy, xx


def detect_color_regions(jpg_on_raw, yy, xx):
    """
    Detect red, green and blue pen regions in the JPG mapped to RAW coordinates.

    The region limit skips background and desk so only the pen area is checked.
    """
    R = jpg_on_raw[:, :, 0]
    G = jpg_on_raw[:, :, 1]
    B = jpg_on_raw[:, :, 2]

    # Only threshold inside the empirically chosen area that contains the pens.
    region = (
        (yy > 1000) & (yy < 3000) &
        (xx > 1500) & (xx < 5200)
    )

    # A colour region is where its main channel clearly dominates the other two.
    red_area = region & (R > 120) & (R > 1.35 * G) & (R > 1.35 * B)
    green_area = region & (G > 80) & (G > 1.20 * R) & (G > 1.20 * B)
    blue_area = region & (B > 80) & (B > 1.25 * R) & (B > 1.15 * G)

    return red_area, green_area, blue_area


def create_boolean_bayer_masks(shape, pattern):
    """
    Create boolean Bayer masks.

    Pattern order:
        pattern[0] pattern[1]
        pattern[2] pattern[3]

    Example:
        GBRG means:
            G B
            R G
    """
    h, w = shape

    # Boolean masks are ANDed directly with the colour region masks.
    r_mask = np.zeros((h, w), dtype=bool)
    g_mask = np.zeros((h, w), dtype=bool)
    b_mask = np.zeros((h, w), dtype=bool)

    positions = [
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    ]

    for row_start, col_start, color in positions:
        # All pixel positions for the current 2x2 phase.
        m = np.zeros((h, w), dtype=bool)
        m[row_start::2, col_start::2] = True

        if color == "R":
            r_mask |= m
        elif color == "G":
            g_mask |= m
        elif color == "B":
            b_mask |= m
        else:
            raise ValueError(f"Unknown Bayer color: {color}")

    return r_mask, g_mask, b_mask


def mean_raw_value(raw, color_mask, area_mask):
    """Mean RAW value of one colour position inside a given image region."""
    mask = color_mask & area_mask

    # Return nan when the colour region was not detected, to avoid false scores.
    if mask.sum() == 0:
        return np.nan

    return raw[mask].mean()


def evaluate_pattern(raw, pattern, red_area, green_area, blue_area):
    """
    Score whether a Bayer pattern is plausible.

    A plausible pattern should satisfy:
        red region: R response above B response
        blue region: B response above R response
        green region: G response comparatively high

    Green is usually brighter in RAW, so red/blue evidence is more reliable.
    """
    r_mask, g_mask, b_mask = create_boolean_bayer_masks(raw.shape, pattern)

    # Mean RAW value for each Bayer colour inside the red, green and blue regions.
    red_R = mean_raw_value(raw, r_mask, red_area)
    red_G = mean_raw_value(raw, g_mask, red_area)
    red_B = mean_raw_value(raw, b_mask, red_area)

    green_R = mean_raw_value(raw, r_mask, green_area)
    green_G = mean_raw_value(raw, g_mask, green_area)
    green_B = mean_raw_value(raw, b_mask, green_area)

    blue_R = mean_raw_value(raw, r_mask, blue_area)
    blue_G = mean_raw_value(raw, g_mask, blue_area)
    blue_B = mean_raw_value(raw, b_mask, blue_area)

    # A higher score means the pattern matches the expected red/blue/green response.
    red_score = red_R - red_B
    blue_score = blue_B - blue_R
    green_score = green_G - max(green_R, green_B)

    total_score = red_score + blue_score + 0.5 * green_score

    return {
        "pattern": pattern,
        "red_R": red_R,
        "red_G": red_G,
        "red_B": red_B,
        "green_R": green_R,
        "green_G": green_G,
        "green_B": green_B,
        "blue_R": blue_R,
        "blue_G": blue_G,
        "blue_B": blue_B,
        "red_score": red_score,
        "blue_score": blue_score,
        "green_score": green_score,
        "total_score": total_score,
    }


def visualize_detected_regions(red_area, green_area, blue_area):
    """Visualize the detected red/green/blue regions to check the thresholds."""
    mask_vis = np.zeros((*red_area.shape, 3), dtype=np.float32)

    # Show the three region types in the RGB channels.
    mask_vis[:, :, 0] = red_area.astype(np.float32)
    mask_vis[:, :, 1] = green_area.astype(np.float32)
    mask_vis[:, :, 2] = blue_area.astype(np.float32)

    plt.figure(figsize=(10, 6))
    plt.imshow(mask_vis)
    plt.title("Automatically detected red / green / blue pen regions")
    plt.axis("off")
    plt.show()


def investigate_bayer_pattern_elegant(raw_path=SAMPLE_NPY, jpg_path=SAMPLE_JPG):
    """Run the full Bayer-pattern detection flow and print the ranking of all four patterns."""
    raw, jpg = load_data(raw_path, jpg_path)

    print("RAW shape:", raw.shape)
    print("JPG shape:", jpg.shape)
    print("RAW min:", raw.min())
    print("RAW max:", raw.max())

    jpg_on_raw, yy, xx = map_jpg_to_raw_coordinates(raw, jpg)

    # Extract the coloured pen regions from the JPG reference.
    red_area, green_area, blue_area = detect_color_regions(jpg_on_raw, yy, xx)

    print("Detected red pixels:", int(red_area.sum()))
    print("Detected green pixels:", int(green_area.sum()))
    print("Detected blue pixels:", int(blue_area.sum()))

    visualize_detected_regions(red_area, green_area, blue_area)

    patterns = ["RGGB", "GRBG", "GBRG", "BGGR"]
    results = []

    for pattern in patterns:
        # Compute the colour response score for each Bayer pattern.
        result = evaluate_pattern(
            raw,
            pattern,
            red_area,
            green_area,
            blue_area
        )
        results.append(result)

        print("\nPattern:", pattern)
        print("Red area means:   R={:.2f}, G={:.2f}, B={:.2f}".format(
            result["red_R"], result["red_G"], result["red_B"]
        ))
        print("Green area means: R={:.2f}, G={:.2f}, B={:.2f}".format(
            result["green_R"], result["green_G"], result["green_B"]
        ))
        print("Blue area means:  R={:.2f}, G={:.2f}, B={:.2f}".format(
            result["blue_R"], result["blue_G"], result["blue_B"]
        ))
        print("red_score   = {:.2f}".format(result["red_score"]))
        print("blue_score  = {:.2f}".format(result["blue_score"]))
        print("green_score = {:.2f}".format(result["green_score"]))
        print("total_score = {:.2f}".format(result["total_score"]))

    print("\nRanking:")
    # Sort by total score, highest first; the top pattern is the estimate.
    ranked = sorted(results, key=lambda x: x["total_score"], reverse=True)

    for result in ranked:
        print("{}: total_score = {:.2f}".format(
            result["pattern"],
            result["total_score"]
        ))

    best = ranked[0]

    print("\nBest Bayer pattern for IMG_9939.npy:", best["pattern"])
    print("Pattern layout:")
    print(best["pattern"][0], best["pattern"][1])
    print(best["pattern"][2], best["pattern"][3])

    return best["pattern"], ranked


if __name__ == "__main__":
    investigate_bayer_pattern_elegant()
