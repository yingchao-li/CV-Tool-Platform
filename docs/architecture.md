# Architecture

CVKit is a single installable package with five bounded subsystems. The design
goal is that each subsystem can be used, tested and understood on its own, while
sharing the same conventions and entry points.

```
                      ┌──────────────────────────────┐
                      │            cvkit             │
                      │  paths · common · cli        │
                      └──────────────┬───────────────┘
                                     │
     ┌──────────────┬────────────────┼────────────────┬──────────────┐
     │              │                │                │              │
 geometry        imaging         retrieval        detection        faces
 point clouds    RAW/JPEG        SIFT + VLAD      proposals        MTCNN + FaceNet
```

## Package layout

```text
src/cvkit/
├── __init__.py          # version, package docstring
├── __main__.py          # python -m cvkit
├── cli.py               # module dispatcher
├── paths.py             # PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, ...
├── common/
│   └── io.py            # JSON / CSV writers
├── geometry/
│   ├── dataset.py       # .mat loading, validity masks, summary
│   ├── planes.py        # (normal, offset) primitives
│   ├── ransac.py        # vanilla RANSAC + RMSE helper
│   ├── mlesac.py        # likelihood-based scoring
│   ├── preemptive.py    # batched hypothesis pruning
│   ├── masks.py         # morphology, connected components
│   ├── measurement.py   # height / length / width, corners
│   ├── render.py        # 2D and 3D overlays
│   ├── pipeline.py      # measure_box() -> BoxMeasurement
│   └── cli.py
├── imaging/
│   ├── config.py        # Bayer phases, default captures
│   ├── bayer.py         # CFA masks + bilinear demosaicing
│   ├── image_ops.py     # RAW loading, tone curves, display helpers
│   ├── pipeline.py      # demosaic -> WB -> tone -> HDR -> iCAM06
│   ├── pattern_detection.py
│   ├── response_io.py   # JPEG bracket discovery and sampling
│   ├── response.py      # shared gamma model and curve output
│   ├── response_gamma.py
│   ├── response_pairwise.py
│   ├── response_quantile.py
│   ├── response_hdr.py  # radiance merge + display mapping
│   └── cli.py
├── retrieval/
│   ├── pipeline.py      # SIFT, codebook, VLAD, E-SVM, evaluation
│   ├── experiments/     # five ablation drivers
│   └── cli.py
├── detection/
│   ├── selective_search.py
│   ├── boxes.py         # IoU, NMS, box conversions
│   ├── datasets.py      # COCO loading, proposal cache paths
│   ├── proposals.py     # proposal generation + cache
│   ├── features.py      # HOG descriptors
│   ├── trainer.py       # IoU-labelled linear SVM
│   ├── detector.py      # threshold + NMS + visualisation
│   ├── evaluation.py    # MABO and mAP
│   ├── tuning.py        # two-stage parameter search
│   ├── pipeline.py      # end-to-end baseline
│   └── cli.py
└── faces/
    ├── config.py        # dataset locations, ReIdMode
    ├── detector.py      # MTCNN detection, tracking, alignment
    ├── recognition.py   # FaceNet, k-NN gallery, k-means clustering
    ├── neighbors.py     # nearest-neighbour classifier
    ├── dir_eval.py      # DIR over similarity thresholds
    ├── openset.py       # SPL / MPL pseudo-labelling
    ├── train.py         # gallery / clustering training CLI
    ├── infer.py         # identification / clustering inference CLI
    ├── camera.py        # webcam tracking demo
    └── cli.py
```

## Conventions

| Concept | Convention |
|---|---|
| Plane | `(normal, offset)` with `normal · x = offset`; normal is unit length |
| Box | `(x1, y1, x2, y2)`, bottom-right exclusive |
| Image | float RGB in `[0, 1]` internally, `uint8` only at export |
| Feature vector | L2-normalised rows before cosine ranking |
| Randomness | every public function accepts `random_seed` / `seed` |
| Paths | never hard-coded; always derived from `cvkit.paths` |

These rules exist because the original algorithms mixed at least three plane
representations and two box conventions. Normalising them removed a whole class
of silent bugs.

## Data flow

### Geometry

```text
.mat -> load_mat_file -> (H,W,3) cloud
      -> valid_points_with_indices
      -> robust plane fit (floor)
      -> mask cleanup -> non-floor points
      -> robust plane fit (box top)
      -> largest component -> PCA sizing
      -> BoxMeasurement
```

`BoxMeasurement.as_dict()` is the serialisation boundary used by the CLI, the
benchmarks and the studio.

### Imaging

```text
CR3 -> rawpy -> 2D Bayer
    -> demosaic -> Gray-World WB -> percentile normalise -> gamma -> enhance
JPEG bracket -> inverse response -> exposure scaling -> radiance merge -> tone map
```

All three response estimators share `response_io` (sampling) and `response`
(model, weights, curve output); only the objective differs.

### Retrieval

```text
images -> RootSIFT -> codebook (k-means)
       -> VLAD residuals -> normalisation -> optional PCA / E-SVM / AQE
       -> cosine ranking -> Top-1, mAP
```

Experiments are independent drivers that reuse the base functions through
`experiments/common.py`, including a parameter-aware cache.

### Detection

```text
image -> Felzenszwalb -> region merging -> proposals -> HOG
      -> linear SVM -> score threshold -> NMS -> detections -> MABO / mAP
```

Proposals are cached per parameter tuple, which is what makes the two-stage
search affordable.

### Faces

```text
frame -> MTCNN -> aligned 224x224 face -> color + gray embeddings
      -> k-NN gallery (identification) or k-means (clustering)
      -> distance / posterior -> accept or reject as unknown
```

The open-set module reuses the same `FaceNet` embeddings and trains SPL/MPL
classifiers on known plus pseudo-unknown classes.

## Extension points

- **New robust estimator**: implement `(points, **params) -> (plane, inliers, score)`
  and add it to `geometry/pipeline.py::_fit`.
- **New response model**: add a module with `run_method(...)` and register it in
  `imaging/cli.py`.
- **New retrieval encoder**: add a function producing an `(N, D)` matrix; reuse
  `pipeline.evaluate`.
- **New detector head**: replace `features.py` + `trainer.py`; proposal caching
  and evaluation stay unchanged.

## Error handling

- Dataset loaders validate shapes eagerly and raise `ValueError` with the actual
  and expected shapes.
- Degenerate geometry (collinear samples, non-finite points) returns `None` at
  the fitting primitive and is skipped by the sampler.
- Detection pipeline steps raise `StepExecutionError`, which keeps the captured
  stdout so a failed run can still be diagnosed from the UI.
- The studio catches per-experiment failures so one broken configuration does
  not take down the whole page.

## Testing strategy

The unit tests target deterministic, dependency-light code: plane conventions,
mask morphology, IoU/NMS, VLAD normalisation and Top-1/mAP. The heavier
pipelines are covered by the benchmark scripts, which are reproducible and write
machine-readable output.
