# Data directory

The toolkit reads every dataset from this folder. Paths are resolved through
`cvkit.paths`, so nothing here is hard-coded in the algorithms.

```text
data/
├── pointcloud/   4 depth-camera captures (.mat)      committed
├── raw/          CR3 / JPG / NPY exposure brackets   not committed (645 MB)
├── balloons/     balloon detection set (COCO)        supplied locally
├── paintings/    9 images for proposal demos         committed
└── faces/        face sequences, galleries, model    supplied locally
```

See [`../docs/data.md`](../docs/data.md) for the full layout, provenance,
licences and regeneration instructions.

Quick checks:

```bash
python -m cvkit geometry --list
python -m cvkit imaging linearity
python -m cvkit detection run
```

If your data lives elsewhere, set `CVKIT_DATA_DIR` before running anything:

```bash
export CVKIT_DATA_DIR=/path/to/cvkit-data
```
