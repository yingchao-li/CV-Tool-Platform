# Benchmarks

Every table on this page is produced by a script under `benchmarks/` or by a
`cvkit` command. Numbers are copied from the reference run shipped with the
repository; re-running the scripts on another machine will change runtimes but
not the qualitative conclusions.

| Artefact | Command |
|---|---|
| `docs/results/geometry_comparison.csv` | `python benchmarks/run_geometry.py` |
| `docs/results/faces/*.csv` | `python benchmarks/run_faces.py` |
| `docs/results/response/*` | `python -m cvkit imaging response --method gamma` |
| `outputs/detection/parameter_search/*` | `python -m cvkit detection tune` |

---

## Geometry

Capture: `data/pointcloud/example1kinect.mat` (H × W × 3 point cloud).
Settings: 1000 iterations, floor ε = 0.05 m, box-top ε = 0.005 m, seed 42.

| Metric | RANSAC | MLESAC | Preemptive RANSAC |
|---|---:|---:|---:|
| Runtime (s) | 5.9625 | 8.2402 | **0.0372** |
| Floor inliers | **156 215** | 154 626 | 154 901 |
| Box-top inliers | 35 938 | **35 954** | 35 799 |
| Floor residual RMSE | 0.018714 | **0.011609** | 0.011909 |
| Box-top residual RMSE | 0.001522 | **0.001445** | 0.001769 |
| Estimated height (m) | 0.1725 | 0.1929 | 0.1981 |
| Estimated length (m) | 0.4580 | 0.4566 | 0.4566 |
| Estimated width (m) | 0.3654 | 0.3624 | 0.3639 |
| Score | 45 792 outliers | 5 897.87 cost | 170 cost |

**Reading the table.** RANSAC maximises the inlier count, so it wins that column
by construction — but the floor plane it finds is 60% worse in RMSE than MLESAC.
MLESAC adds residual information to the objective and produces the tightest
planes at the cost of the longest runtime. Preemptive RANSAC discards half of its
hypotheses after every batch, cutting runtime by ~160× with no visible change in
the segmentation.

The committed CSV contains all four captures (examples 1-4); the table above
reports capture 1, the one used in the figures below.

### MLESAC threshold sensitivity

ε_floor / ε_box = 0.03/0.003 … 0.50/0.050:

| ε_floor / ε_box | Floor inliers | Box inliers | Floor RMSE | Box RMSE | Height (m) | Length (m) | Width (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.03 / 0.003 | 150 585 | 34 635 | 0.0104 | 0.0013 | 0.1881 | 0.4559 | 0.3614 |
| 0.05 / 0.005 | 155 205 | 35 949 | 0.0121 | 0.0014 | 0.1894 | 0.4566 | 0.3629 |
| 0.07 / 0.007 | 157 090 | 36 234 | 0.0137 | 0.0015 | 0.1886 | 0.4596 | 0.3672 |
| 0.10 / 0.010 | 159 225 | 36 527 | 0.0168 | 0.0017 | 0.1893 | 0.4635 | 0.3715 |
| 0.50 / 0.050 | 202 007 | 6 351 | 0.0836 | 0.0134 | 0.0063 | 1.6406 | 0.9474 |

A larger threshold admits more inliers but increases RMSE; the estimated box
stays within 1 mm of 0.189 m until the threshold becomes physically meaningless
(0.50 m).

### Preemptive RANSAC: hypotheses vs. batch size

| M | B | Runtime (s) | Floor RMSE | Height (m) |
|---:|---:|---:|---:|---:|
| 50 | 50 | 0.0259 | 0.0116 | 0.1909 |
| 50 | 200 | 0.0277 | 0.0119 | 0.1955 |
| 200 | 50 | 0.0513 | 0.0116 | 0.1952 |
| 200 | 100 | 0.0412 | 0.0119 | 0.1981 |
| 200 | 200 | 0.0441 | 0.0125 | 0.1896 |
| 500 | 50 | 0.0877 | 0.0192 | 0.1616 |
| 500 | 200 | 0.0688 | 0.0212 | 0.1650 |

Runtime is dominated by the number of hypotheses `M`; increasing `M` past 200
does not improve accuracy and starts to hurt it.

---

## Imaging

### JPEG inverse response

Six bracketed JPEGs (1/10 s … 1/320 s), 50 000 shared pixel samples:

| Estimator | γ | Inverse response | Fitting error |
|---|---:|---|---:|
| Exposure consistency | 1.5537 | `h(y) = y^1.5537` | 0.2066 (weighted relative RMSE) |
| Pairwise exposure ratio | 1.1706 | `h(y) = y^1.1706` | 0.3489 (log-ratio RMSE) |
| Quantile matching | 1.2156 | `h(y) = y^1.2156` | 0.3456 (log-quantile RMSE) |

The three estimators use different statistics over the same samples, so exact
agreement is not expected. The pairwise and quantile estimates agree within
0.05; the consistency estimator is pulled upward by the longest exposure, whose
highlights carry the most weight.

### Sensor linearity

`python -m cvkit imaging linearity` plots mean RAW response against exposure
time for six exposures spanning 32×. The fit is linear over the valid range
(R² > 0.999), which justifies the linear radiance model used by the HDR merge.

### HDR

Two routes are implemented:

- **RAW bracket** — replace saturated pixels with exposure-scaled shorter
  frames, then log or iCAM06 tone-map.
- **JPEG bracket** — linearise each frame with the estimated inverse response,
  then reuse the same replacement merge.

The shared merge function keeps both routes numerically consistent.

---

## Retrieval

Dataset: ICDAR 2017 writer identification (1182 training images, 3600 test
images). RootSIFT, K = 100, VLAD dimension 12 800 unless stated.

| Category | Method | Top-1 | mAP |
|---|---|---:|---:|
| Baseline | Base VLAD | 0.8189 | 0.6294 |
| Baseline | VLAD + power normalisation | 0.8314 | 0.6430 |
| Baseline | VLAD + power normalisation + E-SVM | **0.8947** | **0.7653** |
| Normalisation | + intra-normalisation | 0.8369 | 0.6481 |
| Projection | + PCA whitening 1024-D | 0.8847 | 0.7443 |
| Projection | + PCA whitening 1181-D | 0.8842 | 0.7424 |
| Re-ranking | + AQE k = 5 | 0.7506 | 0.6381 |
| Re-ranking | + AQE k = 10 | 0.6531 | 0.5233 |
| Re-ranking | + AQE k = 20 | 0.5475 | 0.3962 |
| Codebook | K = 64 | 0.8203 | 0.6288 |
| Codebook | K = 100 | 0.8322 | 0.6453 |
| Codebook | K = 128 | 0.8350 | 0.6483 |
| Codebook | K = 200 | 0.8389 | 0.6567 |
| Preprocessing | Image inversion | 0.7472 | 0.5245 |

Notes:

- **Exemplar SVM** is the strongest single change (+0.12 mAP): it re-encodes
  each query using a linear SVM trained against the rest of the gallery.
- **PCA whitening** is the strongest *unsupervised* change (+0.10 mAP at 1024-D).
  The requested 2048/4096-D projections collapse to 1181-D because there are
  only 1182 training images.
- **AQE** assumes the top-k neighbours are mostly correct. In this benchmark the
  premise fails, so every k degrades both metrics.
- **Larger codebooks** help monotonically over the tested range, consistent with
  writer identification depending on fine stroke structure.
- **Image inversion** removes the mid-grey gradients SIFT relies on and loses
  ~0.10 mAP.

---

## Detection

Dataset: balloon set (COCO annotations, 61 images). Proposal cache keyed by
`scale`, `sigma`, `min_size`, `max_proposals`. HOG on 64 × 64 crops, linear SVM
with balanced class weights. mAP uses the 101-point fallback backend for both
rows, so the comparison is apples to apples.

| Experiment | Split | MABO | mAP@[.50:.95] |
|---|---|---:|---:|
| Baseline | test | 0.5567 | 0.1246 |
| Best search trial | valid | 0.6296 | 0.0845 |
| Final configuration | test | **0.6327** | **0.1797** |

Selected parameters:

```text
proposal : scale=100, sigma=1.0, min_size=30, max_proposals=3000
training : tp=0.75, tn=0.25, pos=32, neg=96, neg/pos=5, C=1.0, balanced
inference: score_threshold=-1e6 (no filtering), nms=0.3, max_detections=200
```

The tuned configuration deliberately keeps recall high: it produces ~1070
detections across six test images, which maximises mAP at the cost of precision.
A precision-oriented follow-up should sweep `score_threshold` on validation.

---

## Faces

Dataset: five known identities in `train_data`, two extra identities only in
`test_data` (unknown), 40 known / 20 unknown test faces.

### Tracking

Template matching after an initial MTCNN detection; search window 25 px,
normalised cross-correlation threshold 0.50.

| Metric | Value |
|---|---:|
| Template successes | 33 / 45 frames |
| MTCNN calls | 12 |
| Frames with no face | 11 |
| Average processing time | 0.0125 s/frame |

### Identification and open-set rejection

Best configuration per k and rejection rule (threshold search on validation):

| k | Rule | d threshold | Closed-set rank-1 | Known ID | Known FR | Unknown rejection | Unknown FA |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | distance | 1.05 | 1.0000 | 0.9500 | 0.0500 | **1.0000** | 0.0000 |
| 1 | posterior | 0.00 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 3 | posterior | 0.67 | 1.0000 | 1.0000 | 0.0000 | 0.3000 | 0.7000 |
| 5 | posterior | 1.00 | 1.0000 | 1.0000 | 0.0000 | 0.6000 | 0.4000 |
| 7 | posterior | 1.00 | 1.0000 | 1.0000 | 0.0000 | 0.6500 | 0.3500 |

Distance-only rejection dominates: posterior probabilities saturate at 1.00
for k = 1 and are poorly calibrated for larger k. Combining both rules never
beats distance alone on this dataset.

### Robustness (synthetic transforms, 40 known faces)

| Condition | ID rate | False rejection | Mean distance |
|---|---:|---:|---:|
| Original | 0.9500 | 0.0500 | 0.7709 |
| Darkening | 0.9750 | 0.0250 | 0.7793 |
| Lower-face occlusion | 0.7250 | 0.2750 | 0.9249 |
| Lateral crop | 0.9000 | 0.1000 | 0.8684 |

Occlusion is the dominant failure mode — expected for a global embedding with
no part-based pooling.

### Clustering

k-means, 3 identities × 15 faces, seeds 0/1/2/42/100:

| Seed | Final objective | Iterations | Cluster sizes | Purity |
|---:|---:|---:|---|---:|
| 0 | 14.9827 | 2 | 5 / 10 / 30 | 0.6667 |
| 1 | 15.0514 | 2 | 8 / 7 / 30 | 0.6667 |
| 2 | 4.3950 | 2 | 14 / 16 / 15 | 0.9778 |
| 42 | **4.2887** | 2 | 15 / 14 / 16 | **0.9778** |
| 100 | 4.2887 | 2 | 14 / 15 / 16 | 0.9778 |

Two distinct local optima exist; empty-cluster repair prevents NaNs but cannot
escape a bad initialisation, so multiple seeds are required.

### DIR

| Condition | FAR | Identification rate | Similarity threshold |
|---|---:|---:|---:|
| max ID with FAR ≤ 1% | 0.00998 | 0.7071 | −0.7050 |
| min FAR with ID ≥ 90% | 0.1406 | 0.9001 | −0.8197 |

### Open-set scoring (SPL / MPL)

Five-seed average, 20% of identities held out as unknown:

| Method | AUC | DIR@FAR=1% | DIR@FAR=10% | Balanced Rank-1 | Actual FAR@10% |
|---|---:|---:|---:|---:|---:|
| SPL baseline | 0.9781 | 0.8533 | 0.9402 | 0.9791 | 10.10% |
| MPL baseline | 0.9781 | 0.8529 | 0.9393 | 0.9750 | 9.72% |
| SPL fusion w=1 | 0.9827 | 0.7840 | 0.7840 | 0.7840 | 0.18% |
| MPL fusion w=1 | 0.9814 | 0.8742 | 0.8865 | 0.8865 | 1.40% |

Fusion improves ranking (AUC) but a fixed zero threshold rejects too many known
faces, so the baseline distance margin is retained.
