# Data

CVKit expects a single `data/` directory at the repository root. The location
can be overridden with `CVKIT_DATA_DIR`; outputs go to `outputs/` (override with
`CVKIT_OUTPUT_DIR`).

Large recordings and the face and balloon datasets are **not committed**. The
table below lists what ships with the repository and what must be supplied locally.

```text
data/
├── pointcloud/                 4 × .mat              16 MB   committed
├── raw/                        CR3 / JPEG / NPY      645 MB   not committed
│   ├── 01/  IMG_9939.npy, IMG_9939.JPG
│   ├── 02/  IMG_4782.CR3
│   ├── 05/  IMG_3044…3049.CR3 + JPG      (linearity bracket)
│   └── 06/  00…10.CR3                    (HDR bracket)
├── balloons/                   COCO image set          2 MB   not committed
├── paintings/                  9 JPEGs                3 MB   committed
└── faces/                      1557 JPEGs           185 MB   not committed
    ├── train_data/<identity>/%04d.jpg
    ├── test_data/<identity>/%04d.jpg
    ├── recognition_gallery.pkl
    ├── clustering_gallery.pkl
    ├── evaluation_{train,test}_data.pkl
    └── resnet50_128.onnx
```

## Provenance and licence

| Dataset | Content | Notes |
|---|---|---|
| `pointcloud` | Four depth-camera captures (amplitude, distance, `H×W×3` cloud) | Captured for this project; free to use. |
| `raw` | Canon CR3 RAW files plus exported JPEGs and one NPY dump | Captured for this project; free to use. |
| `balloons` | 61 balloon images with COCO annotations | Third-party dataset; check its original licence before redistribution. |
| `paintings` | Nine painting/architecture images | Used for qualitative proposal demos. |
| `faces` | Short video sequences of seven public figures | Research/benchmark use only; do not redistribute. |
| ICDAR 2017 | Writer-identification images | **Not included**; download from the ICDAR/`ScriptNet` release. |

`data/faces` and `data/balloons` are excluded from this public repository. Supply
your own data locally and check the original dataset licences before redistribution.

## Retrieval dataset

The retrieval module does not assume a path. Pass the four required arguments:

```bash
python -m cvkit retrieval run \
  --in_train  /path/to/icdar2017-training-binary \
  --in_test   /path/to/ScriptNet-HistoricalWI-2017-binarized \
  --labels_train /path/to/icdar17_labels_train.txt \
  --labels_test  /path/to/icdar17_labels_test.txt \
  --powernorm
```

Label files contain one `image_name author_id` pair per line. The loader accepts
`.png` for training images and `.jpg` for test images by default; change
`--suffix_train` / `--suffix_test` if your copy differs.

## Git and large files

`data/raw`, the face fixtures and the ONNX model are excluded by `.gitignore`.
If you need them in the repository, use Git LFS:

```bash
git lfs install
git lfs track "data/raw/**/*.CR3" "data/faces/**/*.jp*g" "data/faces/*.onnx"
git add .gitattributes
```

Alternatively, keep the code repository small and provide a download script for
the recordings.

## Regenerating small fixtures

- **Point clouds** — any capture with `amplitudes*`, `distances*` and `cloud*`
  fields works; the loader only requires the shapes to match.
- **RAW** — the stage functions accept any CR3 together with the correct Bayer
  phase (`RGGB` for the bundled camera).
- **Face galleries** — rebuild them with
  `python -m cvkit faces train --mode ident --label <name> --video ...`; one
  gallery file is extended per identity.
- **Proposal caches** — regenerated automatically on first run and keyed by the
  full proposal parameter tuple.
