"""RAW processing pipeline: previews, white balance, HDR merge and tone mapping."""

import os
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import glob

from scipy.ndimage import gaussian_filter

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .config import PATTERN_CR3, SAMPLE_RAW, EXPOSURE_DIR, BRACKET_DIR
from .image_ops import (
    load_raw_cr3,
    save_float_image_as_jpg,
    percentile_normalize,
    gamma_correction,
    log_curve,
    apply_brightness_contrast, sigmoid_curve,
)
from .bayer import demosaic_simple


# ============================================================
# Demosaicing
# Objective: Read a single CR3 RAW file, perform basic demosaicing,
# and export visualization preview.
# ============================================================

def demosaic_raw(
    raw_path=SAMPLE_RAW,
    output_path="demosaic_rawing_preview.jpg",
    pattern=PATTERN_CR3
):
    # Load raw Bayer data from the camera's RAW.
    raw = load_raw_cr3(raw_path)

    print("Raw shape:", raw.shape)
    print("Raw min/max:", raw.min(), raw.max())

    # Interpolate single-channel RAW into RGB according to the specified Bayer pattern.
    rgb = demosaic_simple(raw, pattern=pattern)

    print("RGB shape:", rgb.shape)
    print("RGB min/max:", rgb.min(), rgb.max())

    # RAW value range is typically much larger than [0, 1],
    # normalize using percentiles before saving.
    rgb_norm, a, b = percentile_normalize(rgb, 0.01, 99.99)
    save_float_image_as_jpg(rgb_norm, output_path, quality=98)

    print("Saved:", output_path)
    return rgb


# ============================================================
# Tone-mapping preview
# Objective: Compare the impact of different brightness mapping curves
# (gamma correction, log curve, etc.) on demosaicing results.
# ============================================================

def preview_tone_curves(
    raw_path=SAMPLE_RAW,
    pattern=PATTERN_CR3
):
    raw = load_raw_cr3(raw_path)
    rgb = demosaic_simple(raw, pattern=pattern)

    # First map RAW RGB to display-friendly [0, 1].
    rgb_norm, a, b = percentile_normalize(rgb, low=0.01, high=99.99)

    # Lower gamma brightens shadows more noticeably.
    rgb_gamma_03 = gamma_correction(rgb_norm, gamma=0.3)
    rgb_gamma_04 = gamma_correction(rgb_norm, gamma=0.4)
    rgb_gamma_05 = gamma_correction(rgb_norm, gamma=0.5)

    # Log curve is another way to compress dynamic range.
    rgb_log = log_curve(rgb_norm, alpha=10.0)
    sigmoid_c=sigmoid_curve(rgb_norm)
    # Save each version separately for visual comparison of different curves.
    save_float_image_as_jpg(rgb_norm, "tone_normalized.jpg", quality=98)
    save_float_image_as_jpg(rgb_gamma_03, "tone_gamma_03.jpg", quality=98)
    save_float_image_as_jpg(rgb_gamma_04, "tone_gamma_04.jpg", quality=98)
    save_float_image_as_jpg(rgb_gamma_05, "tone_gamma_05.jpg", quality=98)
    save_float_image_as_jpg(rgb_log, "tone_log.jpg", quality=98)
    save_float_image_as_jpg(sigmoid_c, "tone_sigmoid.jpg", quality=98)
    print("Saved tone-mapping previews.")
    return rgb_norm, rgb_gamma_03, rgb_log


# ============================================================
# White-balance preview
# Objective: Apply simple white balance using Gray World assumption.
# ============================================================

def gray_world_white_balance(rgb, verbose=True):
    """
    Gray World White Balance.

    Assumes the average color across an image should be close to gray,
    so it scales R/G/B channels to balance their means.
    """
    rgb = rgb.astype(np.float32).copy()

    # Compute the mean intensity and per-channel means.
    mean_image = rgb.mean()
    mean_r = rgb[:, :, 0].mean()
    mean_g = rgb[:, :, 1].mean()
    mean_b = rgb[:, :, 2].mean()

    eps = 1e-8

    scale_r = mean_image / max(mean_r, eps)
    scale_g = mean_image / max(mean_g, eps)
    scale_b = mean_image / max(mean_b, eps)

    if verbose:
        print("\nGray World white balance:")
        print(f"mean image = {mean_image:.4f}")
        print(f"before: R={mean_r:.4f}, G={mean_g:.4f}, B={mean_b:.4f}")
        print(f"scales: R={scale_r:.4f}, G={scale_g:.4f}, B={scale_b:.4f}")

    # Scale each channel by the ratio of image mean to channel mean.
    rgb[:, :, 0] *= scale_r
    rgb[:, :, 1] *= scale_g
    rgb[:, :, 2] *= scale_b

    if verbose:
        print(
            f"after: R={rgb[:, :, 0].mean():.4f}, "
            f"G={rgb[:, :, 1].mean():.4f}, "
            f"B={rgb[:, :, 2].mean():.4f}"
        )

    return rgb


def preview_white_balance(
    raw_path=SAMPLE_RAW,
    pattern=PATTERN_CR3
):
    raw = load_raw_cr3(raw_path)
    rgb = demosaic_simple(raw, pattern=pattern)

    # Tone-mapping preview: normalize and apply brightness curves first.
    rgb_norm, a, b = percentile_normalize(
        rgb,
        low=0.01,
        high=99.99
    )

    rgb_gamma_03 = gamma_correction(rgb_norm, gamma=0.3)
    rgb_gamma_04 = gamma_correction(rgb_norm, gamma=0.4)
    rgb_gamma_05 = gamma_correction(rgb_norm, gamma=0.5)
    rgb_log = log_curve(rgb_norm, alpha=10.0)

    # White-balance preview: apply Gray World white balance as the final step.
    rgb_norm_wb = gray_world_white_balance(rgb_norm)
    rgb_gamma_03_wb = gray_world_white_balance(
        rgb_gamma_03,
        verbose=False
    )
    rgb_gamma_04_wb = gray_world_white_balance(
        rgb_gamma_04,
        verbose=False
    )
    rgb_gamma_05_wb = gray_world_white_balance(
        rgb_gamma_05,
        verbose=False
    )
    rgb_log_wb = gray_world_white_balance(
        rgb_log,
        verbose=False
    )

    # White balance may produce values greater than 1.
    # save_float_image_as_jpg() clips them to [0, 1] before saving.
    save_float_image_as_jpg(
        rgb_norm_wb,
        "preview_white_balance_normalized.jpg",
        quality=98
    )
    save_float_image_as_jpg(
        rgb_gamma_03_wb,
        "preview_white_balance_gamma_03.jpg",
        quality=98
    )
    save_float_image_as_jpg(
        rgb_gamma_04_wb,
        "preview_white_balance_gamma_04.jpg",
        quality=98
    )
    save_float_image_as_jpg(
        rgb_gamma_05_wb,
        "preview_white_balance_gamma_05.jpg",
        quality=98
    )
    save_float_image_as_jpg(
        rgb_log_wb,
        "preview_white_balance_log_curve.jpg",
        quality=98
    )

    print("Saved white-balance previews.")
    return rgb_gamma_04_wb
# ============================================================
# Sensor linearity
# Objective: Use different exposures in RAW to verify sensor's
# approximately linear response.
# ============================================================

def plot_sensor_linearity(
    folder=EXPOSURE_DIR,
    output_plot="sensor_linearity.png"
):
    # File names are ordered from longest to shortest exposure.
    file_names = [
        "IMG_3044.CR3",
        "IMG_3045.CR3",
        "IMG_3046.CR3",
        "IMG_3047.CR3",
        "IMG_3048.CR3",
        "IMG_3049.CR3",
    ]

    # Exposure times (seconds) corresponding to file names above.
    exposure_times = np.array([
        1 / 10,
        1 / 20,
        1 / 40,
        1 / 80,
        1 / 160,
        1 / 320,
    ], dtype=np.float32)

    mean_values = []

    for index, file_name in enumerate(file_names):
        path = os.path.join(folder, file_name)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # In linear RAW domain, the full-image mean should be roughly
        # proportional to exposure time.
        raw = load_raw_cr3(path)
        mean_value = raw.mean()
        mean_values.append(mean_value)

        print(
            f"{file_name}: "
            f"exposure={exposure_times[index]:.6f}, "
            f"mean={mean_value:.4f}"
        )

    mean_values = np.array(mean_values, dtype=np.float32)

    # Plot the relationship between exposure time and RAW mean value.
    plt.figure(figsize=(8, 5))
    plt.plot(exposure_times, mean_values, marker="o")
    plt.xlabel("Exposure time [s]")
    plt.ylabel("Average raw sensor value")
    plt.title("Sensor response is approximately linear across exposures")
    plt.grid(True)
    plt.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.show()

    print("Saved:", output_plot)
    return exposure_times, mean_values


def run_linearity_experiment():
    data_folder = EXPOSURE_DIR

    # Image files and corresponding exposure times (in seconds)
    images = [
        ('IMG_3044.CR3', 1/10),    # 1/10 seconds
        ('IMG_3045.CR3', 1/20),    # 1/20 seconds
        ('IMG_3046.CR3', 1/40),    # 1/40 seconds
        ('IMG_3047.CR3', 1/80),    # 1/80 seconds
        ('IMG_3048.CR3', 1/160),   # 1/160 seconds
        ('IMG_3049.CR3', 1/320),   # 1/320 seconds
    ]

    exposure_times = []
    average_values = []

    print("Processing images...")
    for filename, exp_time in images:
        filepath = os.path.join(data_folder, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue

        # Load raw data
        raw = rawpy.imread(filepath)
        raw_data = raw.raw_image_visible.astype(np.float32)

        # Compute mean of the entire raw data (not channel-wise)
        mean_val = raw_data.mean()

        exposure_times.append(exp_time)
        average_values.append(mean_val)

        print(f"  {filename}: exposure={exp_time:.6f}s, avg_raw={mean_val:.2f}")

    # Plot
    exposure_times = np.array(exposure_times)
    average_values = np.array(average_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(exposure_times, average_values, s=100, color='blue', label='Measured data')

    # Fit a line to verify linearity
    z = np.polyfit(exposure_times, average_values, 1)
    p = np.poly1d(z)
    fit_line = p(exposure_times)
    ax.plot(exposure_times, fit_line, 'r--', linewidth=2, label=f'Linear fit: y={z[0]:.1f}x+{z[1]:.1f}')

    # Calculate R^2 to show goodness of fit
    residuals = average_values - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((average_values - average_values.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)

    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('Average Raw Data Value', fontsize=12)
    ax.set_title('Sensor Linearity: Average Raw Value vs Exposure Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add R^2 to plot
    textstr = f'$R^2 = {r_squared:.4f}$'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(base, 'ex5_linearity_plot.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")

    # Display plot
    plt.show()

    print("\nLinearity Analysis:")
    print(f"  Slope: {z[0]:.2f} (raw value per second)")
    print(f"  Intercept: {z[1]:.2f}")
    print(f"  R² (goodness of fit): {r_squared:.4f}")
    print(f"  Linear relationship confirmed: {r_squared > 0.99}")

# ============================================================
# HDR radiance merge
# Objective: merge a series of bracketed exposures into an HDR RAW image
# with the threshold-replacement strategy.
# ============================================================


def replace_saturated_values(
    hdr_values,
    current_sensor_values,
    next_values,
    next_sensor_values,
    scale_factor,
    threshold,
):
    """Core merge step: replace saturated values with a scaled shorter exposure.

    ``hdr_values`` may be a 2-D RAW array or an RGB array. The saturation mask is
    always computed from the unscaled sensor/luminance values. Updating
    ``current_sensor_values`` lets a pixel that is still saturated be replaced
    again by the following, shorter exposure.

    Replace saturated pixels with a scaled shorter exposure and keep the
    unscaled values for the next round's threshold check.
    """
    hdr_values = np.asarray(hdr_values)
    current_sensor_values = np.asarray(current_sensor_values)
    next_values = np.asarray(next_values)
    next_sensor_values = np.asarray(next_sensor_values)

    mask_shape = current_sensor_values.shape
    if hdr_values.shape[:len(mask_shape)] != mask_shape:
        raise ValueError("HDR values and sensor mask dimensions do not match")
    if next_values.shape != hdr_values.shape:
        raise ValueError("Current and next exposure values must have matching shapes")
    if next_sensor_values.shape != mask_shape:
        raise ValueError("Current and next sensor values must have matching shapes")

    replace_mask = current_sensor_values > threshold
    scaled_next = next_values * np.float32(scale_factor)
    hdr_values[replace_mask] = scaled_next[replace_mask]
    current_sensor_values[replace_mask] = next_sensor_values[replace_mask]
    return int(np.count_nonzero(replace_mask))


def run_hdr_merge(folder=BRACKET_DIR, output_path="hdr_merged.png"):
    """Merge the RAW bracket, tone-map it and write a JPEG preview."""
    data_dir = folder
    file_pattern = os.path.join(data_dir, "*.CR3")
    files = sorted(glob.glob(file_pattern))

    if not files:
        raise FileNotFoundError(f"No .CR3 files found in '{data_dir}'")

    # Load the brightest raw data (longest exposure) 00.CR3
    with rawpy.imread(files[0]) as raw_0:
        # Extract raw CFA data as float32 to prevent overflow during HDR multiplication
        # 2. Call it h
        h = raw_0.raw_image.astype(np.float32).copy()

        # Keep camera white balance for post-demosaicing [Red, Green, Blue, Green2]
        wb = raw_0.camera_whitebalance

    # 3. Set the threshold t
    # A value of 0.8 * max(h) should be fine
    t = 0.8 * np.max(h)

    # We track the unscaled sensor values to evaluate the threshold in future iterations.
    # This prevents evaluating the threshold against already-multiplied HDR values
    # and safely avoids the 'plateau' (saturation) in subsequent shorter exposures.
    current_sensor_values = h.copy()

    # Loop through each next raw file:
    for idx, file_path in enumerate(files[1:], start=1):
        with rawpy.imread(file_path) as raw_i:
            # Load it and name it i
            i = raw_i.raw_image.astype(np.float32)

            # Multiply i by the exposure difference to the first photo
            # Assuming exposures halve each time: 0th=1x, 1st=2x, 2nd=4x, 3rd=8x...
            multiplier = 2 ** idx
            replace_saturated_values(
                h,
                current_sensor_values,
                i,
                i,
                scale_factor=multiplier,
                threshold=t,
            )

    print("HDR Raw merge complete. Processing image...")

    # --- Post-Processing ---
    print("HDR Raw merge complete. Processing image...")

    # Map float32 data linearly to 16-bit integer range to meet OpenCV requirements.
    h_max = np.max(h)

    h_16bit = np.clip((h / h_max) * 65535.0, 0, 65535).astype(np.uint16)

    # 4. Apply the demosaicing algorithm
    # Now the input is valid 16-bit data
    hdr_bgr_16bit = cv2.cvtColor(h_16bit, cv2.COLOR_BayerBG2BGR)

    # After demosaicing, convert back to float32 to continue with white balance and log operation.
    hdr_bgr = hdr_bgr_16bit.astype(np.float32)

    # 5. Apply the white balance
    # wb array contains [R_scale, G_scale, B_scale, G2_scale]
    b, g, r = cv2.split(hdr_bgr)
    b = b * wb[2]  # Blue channel multiplier
    g = g * wb[1]  # Green channel multiplier
    r = r * wb[0]  # Red channel multiplier
    hdr_wb = cv2.merge((b, g, r))

    # 6. Decrease the dynamic range by computing the logarithm of this data
    # Add 1e-6 to avoid taking log of 0, which results in -inf
    hdr_log = np.log(hdr_wb + 1e-6)

    # 7. Downscale it to the [0, 255] interval
    log_min = np.min(hdr_log)
    log_max = np.max(hdr_log)
    hdr_normalized = (hdr_log - log_min) / (log_max - log_min) * 255.0

    # 8. Save the resulting image
    hdr_final = hdr_normalized.astype(np.uint8)
    cv2.imwrite(output_path, hdr_final)

    print(f"HDR image saved successfully as {output_path}")

    # Return the white balance adjusted HDR image before log compression
    # (reused by the iCAM06 tone-mapping stage)
    return hdr_wb


def load_hdr_raw_paths(folder=BRACKET_DIR):
    """Load HDR bracketed exposure file paths in order from 00.CR3 to 10.CR3."""
    paths = []

    for k in range(11):
        path = os.path.join(folder, f"{k:02d}.CR3")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        paths.append(path)

    return paths


def merge_hdr_raw(folder=BRACKET_DIR, threshold_ratio=0.8):
    paths = load_hdr_raw_paths(folder)

    # Start with the first image as the HDR estimate h.
    h = load_raw_cr3(paths[0]).astype(np.float32)

    # Pixels above this threshold are considered near saturation and need replacement
    # with shorter exposures.
    threshold = threshold_ratio * h.max()
    current_sensor_values = h.copy()

    print("Initial h:", paths[0])
    print("h min/max:", h.min(), h.max())
    print("threshold:", threshold)

    for k in range(1, len(paths)):
        i = load_raw_cr3(paths[k]).astype(np.float32)

        # Assume each subsequent exposure is half: 0th=1x, 1st=2x, 2nd=4x, etc.
        # So shorter exposures need multiplying by 2^k to align to the first exposure.
        scale_factor = 2 ** k
        num_replaced = replace_saturated_values(
            h,
            current_sensor_values,
            i,
            i,
            scale_factor=scale_factor,
            threshold=threshold,
        )

        print(
            f"{os.path.basename(paths[k])}: "
            f"scale={scale_factor}, replaced={num_replaced}"
        )

    print("HDR raw min/max:", h.min(), h.max())
    return h


def log_tone_mapping(rgb):
    """Apply simple log tone mapping to HDR RGB to compress dynamic range for JPG display."""
    rgb = np.maximum(rgb, 0.0)
    log_rgb = np.log(rgb + 1.0)
    log_rgb_norm, a, b = percentile_normalize(log_rgb, low=0.01, high=99.99)
    return log_rgb_norm


def hdr_merge_demo(
    folder=BRACKET_DIR,
    output_path="hdr_log_result.jpg",
    pattern=PATTERN_CR3
):
    # After HDR merging, the result is still Bayer RAW, needs demosaicing first,
    # then white balance and tone mapping.
    hdr_raw = merge_hdr_raw(folder=folder, threshold_ratio=0.8)

    hdr_rgb = demosaic_simple(hdr_raw, pattern=pattern)
    hdr_rgb_wb = gray_world_white_balance(hdr_rgb)

    hdr_log = log_tone_mapping(hdr_rgb_wb)

    save_float_image_as_jpg(hdr_log, output_path, quality=98)
    print("Saved:", output_path)

    return hdr_raw, hdr_rgb_wb, hdr_log


# ============================================================
# iCAM06 tone mapping
# Objective: Implement a simplified iCAM06-based local tone mapping.
# ============================================================

def bilateral_filter_image(img, diameter=9, sigma_color=0.4, sigma_space=9):
    """Prefer OpenCV bilateral filtering; fall back to gaussian filtering if cv2 unavailable."""
    img = img.astype(np.float32)

    if HAS_CV2:
        return cv2.bilateralFilter(
            img,
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

    print("Warning: cv2 not installed, using gaussian_filter instead.")
    return gaussian_filter(img, sigma=2.0)


def icam06_tone_mapping(
    rgb,
    output_range=4.0,
    bilateral_diameter=9,
    sigma_color=0.4,
    sigma_space=9
):
    """Simplified iCAM06: separate luminance into base and detail layers,
    compress the base layer, then restore color ratios.
    """
    eps = 1e-8

    rgb = rgb.astype(np.float32)
    rgb = np.maximum(rgb, 0.0)

    # Estimate input intensity using a green-weighted formula.
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    input_intensity = (20.0 * red + 40.0 * green + blue) / 61.0
    input_intensity = np.maximum(input_intensity, eps)

    r_ratio = red / input_intensity
    g_ratio = green / input_intensity
    b_ratio = blue / input_intensity

    # In log domain, separate base layer and detail layer.
    # Better suited for handling HDR dynamic range.
    log_intensity = np.log(input_intensity)

    log_base = bilateral_filter_image(
        log_intensity,
        diameter=bilateral_diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space
    )

    log_details = log_intensity - log_base

    # Compress the base layer to a specified output dynamic range.
    base_min = log_base.min()
    base_max = log_base.max()

    compression = np.log(output_range) / max(base_max - base_min, eps)
    log_offset = -base_max * compression

    output_intensity = np.exp(
        log_base * compression + log_offset + log_details
    )

    # Maintain original color ratios; only replace the compressed luminance.
    out_r = r_ratio * output_intensity
    out_g = g_ratio * output_intensity
    out_b = b_ratio * output_intensity

    out_rgb = np.stack([out_r, out_g, out_b], axis=2)

    out_rgb_norm, a, b = percentile_normalize(out_rgb, low=0.01, high=99.99)
    return out_rgb_norm


def icam06_demo(
    folder=BRACKET_DIR,
    output_path="icam06_demo_result.jpg",
    pattern=PATTERN_CR3
):
    # iCAM06 reuses the radiance map produced by the HDR merge.
    hdr_raw = merge_hdr_raw(folder=folder, threshold_ratio=0.8)

    hdr_rgb = demosaic_simple(hdr_raw, pattern=pattern)
    hdr_rgb_wb = gray_world_white_balance(hdr_rgb)

    icam_img = icam06_tone_mapping(
        hdr_rgb_wb,
        output_range=4.0,
        bilateral_diameter=9,
        sigma_color=0.4,
        sigma_space=9
    )

    save_float_image_as_jpg(icam_img, output_path, quality=98)
    print("Saved:", output_path)

    return icam_img

def apply_icam06(hdr_bgr_float, output_range=4.0, d=5, sigma_color=0.5, sigma_space=2.0):
    """
    Implement iCAM06 HDR tone mapping algorithm.

    :param hdr_bgr_float: Input HDR image (float32, uncompressed linear RGB)
    :param output_range: Compression ratio for dynamic range (tunable)
    :param d: Bilateral filter kernel size; set small (e.g. 3 or 5) during dev for speed
    :param sigma_color: Sigma for color space of bilateral filter (tunable)
    :param sigma_space: Sigma for coordinate space of bilateral filter (tunable)
    """
    print(f"Running iCAM06 Tone Mapping... (d={d}, output_range={output_range})")

    # Extract BGR channels (OpenCV default format)
    b, g, r = cv2.split(hdr_bgr_float)

    # Define a very small epsilon to prevent division by zero or log(0) errors.
    eps = 1e-6

    # 1. input_intensity = 1/61 * (20*red + 40*green + blue)
    input_intensity = (20.0 * r + 40.0 * g + 1.0 * b) / 61.0
    input_intensity = np.clip(input_intensity, eps, None)

    # 2. r, g, b = rgb / input_intensity (extract color coefficients / Chromaticity)
    r_chroma = r / input_intensity
    g_chroma = g / input_intensity
    b_chroma = b / input_intensity

    # 3. log_base = bilat_filt(log(input_intensity))
    log_intensity = np.log(input_intensity)
    # OpenCV bilateral filter requires float32 data type
    log_intensity_f32 = log_intensity.astype(np.float32)
    # Use bilateral filter to extract base layer (Base Layer)
    log_base = cv2.bilateralFilter(log_intensity_f32, d, sigma_color, sigma_space)

    # 4. log_details = log(input_intensity) - log_base (extract detail layer)
    log_details = log_intensity - log_base

    # 5. compression = log(output_range) / (max(log_base) - min(log_base))
    base_max = np.max(log_base)
    base_min = np.min(log_base)

    range_diff = base_max - base_min
    if range_diff < eps:  # Prevent division by zero for solid color images
        range_diff = eps

    compression = np.log(output_range) / range_diff

    # 6. log_offset = -max(log_base) * compression
    # This step aligns the maximum of the compressed base layer to 0, since exp(0) = 1
    log_offset = -base_max * compression

    # 7. output_intensity = exp(log_base * compression + log_offset + log_detail)
    # Recombine: compressed base layer + offset + preserved detail layer, then exponentiate
    output_intensity = np.exp(log_base * compression + log_offset + log_details)

    # 8. rgb = r*output_intensity, g*output_intensity, b*output_intensity
    r_out = r_chroma * output_intensity
    g_out = g_chroma * output_intensity
    b_out = b_chroma * output_intensity

    # Merge channels
    output_bgr = cv2.merge((b_out, g_out, r_out))

    # Due to the effect of log_offset, the brightest areas are roughly 1.0.
    # Clip to [0.0, 1.0] and convert to uint8 [0, 255] for saving.
    output_bgr = np.clip(output_bgr, 0.0, 1.0)
    final_image = (output_bgr * 255.0).astype(np.uint8)

    return final_image

# ============================================================
# RAW to JPEG pipeline
# Objective: Encapsulate the complete RAW -> JPG processing pipeline.
# ============================================================

def enhance_for_process_raw(
    rgb_norm,
    brightness=0.10,
    contrast=1.75,
    saturation=1.25,
):
    """
    Final contrast and saturation enhancement.

    Simulates manual image adjustment: increase brightness, enhance contrast,
    and slightly boost saturation.
    """
    img = np.clip(rgb_norm, 0.0, 1.0).astype(np.float32)

    # Brighten the overall image.
    img = img + brightness
    img = np.clip(img, 0.0, 1.0)

    # Enhance contrast centered at 0.5.
    img = (img - 0.5) * contrast + 0.5
    img = np.clip(img, 0.0, 1.0)

    # Boost saturation by interpolating between gray and original image.
    gray = img.mean(axis=2, keepdims=True)
    img = gray + saturation * (img - gray)
    img = np.clip(img, 0.0, 1.0)

    return img


def process_raw(
    raw_path,
    jpg_path,
    pattern=PATTERN_CR3,
    black_percentile=0.01,
    norm_low=0.01,
    norm_high=99.99,
    gamma=0.38,
    enhance_brightness=0.10,
    enhance_contrast=1.75,
    enhance_saturation=1.25,
    final_brightness=0.0,
    final_contrast=1.0,
    jpg_quality=99,
    verbose=True,
):
    """
    Full RAW-to-JPEG conversion.

    Processing pipeline:
        CR3 RAW
        -> Black level subtraction
        -> Simple demosaicing
        -> Gray World white balance
        -> Percentile normalization
        -> Gamma correction
        -> Brightness/contrast/saturation enhancement
        -> Optional: final display brightness/contrast
        -> Save JPG
    """
    raw = load_raw_cr3(raw_path)
    rgb_final = process_bayer_raw_to_rgb(
        raw,
        pattern=pattern,
        black_percentile=black_percentile,
        norm_low=norm_low,
        norm_high=norm_high,
        gamma=gamma,
        enhance_brightness=enhance_brightness,
        enhance_contrast=enhance_contrast,
        enhance_saturation=enhance_saturation,
        final_brightness=final_brightness,
        final_contrast=final_contrast,
        verbose=verbose,
    )

    save_float_image_as_jpg(rgb_final, jpg_path, quality=jpg_quality)

    if verbose:
        print("Saved processed JPG:", jpg_path)
    return rgb_final


def process_raw_to_array(
    raw_path,
    pattern=PATTERN_CR3,
    black_percentile=0.01,
    norm_low=0.01,
    norm_high=99.99,
    gamma=0.38,
    enhance_brightness=0.10,
    enhance_contrast=1.75,
    enhance_saturation=1.25,
    final_brightness=0.0,
    final_contrast=1.0,
    verbose=False,
):
    """
    Same pipeline as process_raw, but does not write to disk;
    for use in interactive interfaces (e.g., Streamlit).
    """
    raw = load_raw_cr3(raw_path)
    return process_bayer_raw_to_rgb(
        raw,
        pattern=pattern,
        black_percentile=black_percentile,
        norm_low=norm_low,
        norm_high=norm_high,
        gamma=gamma,
        enhance_brightness=enhance_brightness,
        enhance_contrast=enhance_contrast,
        enhance_saturation=enhance_saturation,
        final_brightness=final_brightness,
        final_contrast=final_contrast,
        verbose=verbose,
    )


def process_bayer_raw_to_rgb(
    raw,
    pattern=PATTERN_CR3,
    black_percentile=0.01,
    norm_low=0.01,
    norm_high=99.99,
    gamma=0.38,
    enhance_brightness=0.10,
    enhance_contrast=1.75,
    enhance_saturation=1.25,
    final_brightness=0.0,
    final_contrast=1.0,
    verbose=False,
):
    """
    Run the complete RAW-to-JPEG pipeline on an already-loaded 2D Bayer RAW (float32),
    returning RGB float32.
    """
    raw = np.array(raw, dtype=np.float32, copy=True)

    black = np.percentile(raw, black_percentile)
    raw = raw - black
    raw = np.clip(raw, 0.0, None)

    rgb = demosaic_simple(raw, pattern=pattern)
    rgb_wb = gray_world_white_balance(rgb, verbose=verbose)

    rgb_wb = np.nan_to_num(rgb_wb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_wb = np.clip(rgb_wb, 0.0, None)

    rgb_norm, a, b = percentile_normalize(rgb_wb, low=norm_low, high=norm_high)
    rgb_gamma = gamma_correction(rgb_norm, gamma=gamma)
    rgb_final = enhance_for_process_raw(
        rgb_gamma,
        brightness=enhance_brightness,
        contrast=enhance_contrast,
        saturation=enhance_saturation,
    )
    return apply_brightness_contrast(
        rgb_final, brightness=final_brightness, contrast=final_contrast
    )


def compare_bayer_patterns(
    raw_path=SAMPLE_RAW
):
    """Render four Bayer arrangements of the same RAW to pick the correct phase visually."""
    patterns = ["RGGB", "GRBG", "GBRG", "BGGR"]

    for pattern in patterns:
        output_path = f"final_test_pattern_{pattern}.jpg"
        print("\n=================================")
        print("Testing pattern:", pattern)
        print("=================================")
        process_raw(raw_path, output_path, pattern=pattern)
        print("Saved:", output_path)
