"""Visual demo: Selective Search proposals on the bundled painting set.

Saves one overlay per input image under ``outputs/detection/proposals``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import skimage.io

from cvkit.detection.selective_search import selective_search
from cvkit.paths import OUTPUT_DETECTION, PAINTING_DIR

OUTPUT_DIR = OUTPUT_DETECTION / "proposals"
MAX_ASPECT_RATIO = 1.2
MIN_REGION_SIZE = 2000


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(PAINTING_DIR.rglob("*.jpg")):
        image = skimage.io.imread(image_path)
        _, regions = selective_search(image, scale=500, min_size=20)

        candidates = set()
        for region in regions:
            if region["size"] < MIN_REGION_SIZE:
                continue
            x, y, w, h = region["rect"]
            if w / h > MAX_ASPECT_RATIO or h / w > MAX_ASPECT_RATIO:
                continue
            candidates.add(region["rect"])

        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image)
        for x, y, w, h in candidates:
            axis.add_patch(
                mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor="red", linewidth=1
                )
            )
        axis.set_title(f"{image_path.name} - {len(candidates)} proposals")
        axis.axis("off")

        relative = image_path.relative_to(PAINTING_DIR)
        output_path = OUTPUT_DIR / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", dpi=120)
        plt.close(figure)
        print(f"{relative}: {len(candidates)} proposals -> {output_path}")


if __name__ == "__main__":
    main()
