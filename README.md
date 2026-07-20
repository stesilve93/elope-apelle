# ELOPE

ELOPE is a PyTorch codebase for estimating lunar-lander velocity from event-camera
streams, IMU telemetry, range-meter measurements, and attitude information. The
repository contains model definitions, dataset loaders, training scripts,
evaluation utilities, and analysis tools used to inspect learned latent spaces.

The project is organized around YAML configuration files for the dataset and the
model. The main scripts live at the repository root, while exploratory and
paper-support analysis scripts live in `analysis/`.

## Repository Layout

```text
elope/              Python package with datasets, models, trainers, and utilities
cfg/dataset/        Dataset preprocessing and loading configurations
cfg/training/       Model and training configurations
best-model/         Example exported configuration for a selected model
analysis/           Latent-space, paper-analysis, event, and flow inspection scripts
docs/               Additional model and analysis notes
LICENSE            MIT license for the repository
train.py            Single split training entry point
train_cross.py      Cross-validation training entry point
test.py             Validation/evaluation entry point
submit.py           Submission JSON generation entry point
env_min.yml         Minimal Conda environment
env_elope.yml       Full exported Conda environment
```

Local data, checkpoints, generated plots, cached datasets, and submission JSON
files are intentionally ignored by Git through `.gitignore`.

## Setup

Create the recommended Conda environment:

```bash
conda env create -f env_min.yml
conda activate elope
```

`env_elope.yml` is a fuller exported environment and can be useful when exact
reproduction of the original development machine is needed.

## Data Layout

Place the ELOPE data under `elope_data/`:

```text
elope_data/
  train/
    0000.npz
    ...
  test/
    0000.npz
    ...
```

Dataset YAML files in `cfg/dataset/` define the source path, cache path,
sequence type, sampling interval, event tensor size, event encoding method, and
event integration window. Cached datasets are written under `dataset/` when
`save_cache: True` is enabled.

## Training

Edit the constants at the top of `train.py` to select the dataset config, model
config, and validation sequences:

```python
DATASET_CFG = "cfg/dataset/dataset-fix-03-last.yml"
MODEL_CFG = "cfg/training/emmnet-angles-of.yml"
SEQUENCE_VAL = [4]
```

Then run:

```bash
python train.py
```

Training outputs are written under `weights/<model-name>_<timestamp>/`. Each
run stores the copied dataset/model configs, checkpoint files, the best
checkpoint, training plots, and optional latent logs.

For cross-validation training, edit the constants at the top of `train_cross.py`
and run:

```bash
python train_cross.py
```

## Evaluation

Edit `MODEL_PATH`, validation sequence settings, and plotting options at the top
of `test.py`, then run:

```bash
python test.py
```

The script loads `model-cfg.yml`, `dataset-cfg.yml`, and `best.pth` from the
selected model folder, evaluates velocity predictions, and writes diagnostic
plots under `plots/testing/` when enabled.

## Submission Generation

Edit `SUBMISSION_NAME`, `CROSS_TRAIN`, and related constants at the top of
`submit.py`, then run:

```bash
python submit.py
```

The script runs inference on `elope_data/test/`, interpolates predictions to the
trajectory timestamps, optionally replaces vertical velocity using the geometric
range-meter constraint, and writes a submission JSON file.

## Analysis Utilities

Analysis and inspection scripts are kept in `analysis/`:

- Latent-space comparison and manifold visualization.
- Classical-behavior probes for learned representations.
- Event and optical-flow visualization helpers.
- Dataset sequence plotting utilities.

Run them from the repository root, for example:

```bash
python analysis/compare_latent_spaces.py --help
python analysis/visualize_latent_manifold.py --help
python analysis/inspect_latents_angles.py --help
```

See `analysis/README.md` for a short map of the available scripts.

## Configuration Notes

- Model variants are selected with the `model` field in `cfg/training/*.yml`.
- Event encodings are selected with `events.encoder_method` in
  `cfg/dataset/*.yml`.
- Most root scripts currently use top-of-file constants rather than command-line
  arguments. Adjust those constants before running a different experiment.
- Checkpoints are expected in `weights/`, which is ignored by Git. Keep public
  model artifacts in a release, model registry, or external storage location if
  they should be shared.

## License

This repository is released under the MIT License. See `LICENSE`.

The license applies to the source code and repository documentation. Dataset
files, trained checkpoints, generated outputs, and third-party assets are not
included unless explicitly stated.

## Public Release Checklist

- Document where the dataset can be obtained, if redistribution is restricted.
- Publish model checkpoints separately if they are too large for Git.
- Confirm that generated files under `weights/`, `plots/`, `dataset/`, and
  `elope_data/` are not staged.

## References
[Link to paper](https://arxiv.org/html/2607.15794v1) <br>
Silvestrini, S. and Ceresoli, M. (2026). *On the Geometry of Learned Representations in Event-Based Egomotion Estimation*. arXiv preprint arXiv:2607.15794. 

