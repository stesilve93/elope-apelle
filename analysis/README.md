# Analysis

This directory contains offline analysis, diagnostics, visualizations, and
paper-support tooling. Training, evaluation, and submission entry points remain
at the repository root.

Run scripts from the repository root. Several defaults intentionally resolve
relative to `elope_data/`, `weights/`, `plots/`, or `analysis/outputs/`.

## Directory Map

```text
analysis/
├── latent/                  latent-space comparison and interpretation
│   └── revision/           held-out revision and probe validation
├── flow_visualization/     event-camera and optical-flow rendering
├── dataset/                raw dataset inspection
├── performance/            runtime benchmarks
├── publishing/             manuscript regeneration helpers
└── outputs/                generated analysis artifacts
    ├── latent/
    ├── revision/
    ├── flow/
    ├── dataset/
    └── paper/
```

The Python directories are packages so scripts can share helpers through stable
imports. Generated files belong under `outputs/`; most binary and data formats
there are already ignored by the repository-wide `.gitignore`.

## Latent Analysis

| Script | Purpose | Main output |
| --- | --- | --- |
| `latent/compare_latent_spaces.py` | Extract, align, and compare with-flow and without-flow representations. | Extracted NPZ packs, JSON metrics, and comparison plots under `--out`. |
| `latent/visualize_latent_manifold.py` | Build PCA, t-SNE, or UMAP manifold views from an extracted pack. | Static PNGs, `summary.json`, and optional interactive HTML under `--out`. |
| `latent/inspect_latents_angles.py` | Inspect velocity decodability, latent correlations, and attention. | Metrics text and diagnostic PNGs under `--out`. |
| `latent/inspect_classical_hints.py` | Probe latents for motion, error, event-density, and trajectory cues. | JSON metrics and PNG plots under `--out`. |
| `latent/analyze_journal_latents.py` | Produce the full journal-oriented reliability, robustness, navigation-gate, attention, and manifold analysis. | CSV tables, figures, arrays, and metadata under `--out-dir`. |

Typical commands:

```bash
python analysis/latent/compare_latent_spaces.py \
  --flow-model-dir weights/emmnet-angles-of_YYYYMMDD_HHMMSS \
  --noflow-model-dir weights/emmnet-angles_YYYYMMDD_HHMMSS \
  --out analysis/outputs/latent/comparison

python analysis/latent/visualize_latent_manifold.py \
  --npz analysis/outputs/latent/comparison/extracted_with_flow.npz \
  --out analysis/outputs/latent/manifold \
  --interactive

python analysis/latent/analyze_journal_latents.py \
  --with-flow-dir plots/latent_compare_best_3 \
  --without-flow-dir plots/latent_compare_best_3 \
  --out-dir analysis/outputs/latent/journal
```

## Revision Validation

- `latent/revision/recompute_revision_validation.py` uses the fixed train and
  validation split to regenerate or validate latent packs and held-out metrics.
  It writes to `outputs/revision/latents/` and
  `outputs/revision/validation_analysis/`.
- `latent/revision/diagnose_revision_probes.py` analyzes component- and
  trajectory-level calibration of the frozen ridge probes. It reads the revision
  packs and writes to `outputs/revision/validation_analysis/probe_diagnostic/`.

```bash
python analysis/latent/revision/recompute_revision_validation.py --reuse-artifacts
python analysis/latent/revision/diagnose_revision_probes.py
```

## Event and Flow Visualization

| Script | Purpose | Main output |
| --- | --- | --- |
| `flow_visualization/visualize_event_flow_pipeline.py` | Render one publication-quality event-tensorization and learned-flow example. | One PNG selected by `--out`; defaults to `outputs/flow/event_flow_pipeline.png`. |
| `flow_visualization/inspect_flow_preds.py` | Compare learned flow with optional EVFlowNet predictions across sequences. | Per-sequence GIFs under `outputs/flow/sequence_flow_preds/`. |
| `flow_visualization/inspect_events.py` | Render binned event surfaces with optional EVFlowNet output. | Per-sequence GIFs under `outputs/flow/sequence_events/`. |
| `flow_visualization/test_evflownet.py` | Run the legacy EVFlowNet smoke test on sequence 0023. | GIF and PNG diagnostics under `outputs/flow/evflownet_smoke_test/`. |

The last three scripts use module-level configuration rather than CLI arguments.
Review their checkpoint, dataset, sequence, and output constants before running
them. `inspect_events.py` and `test_evflownet.py` execute at import time and are
intended to be launched as scripts, not imported as libraries.

Example:

```bash
python analysis/flow_visualization/visualize_event_flow_pipeline.py \
  --model-dir weights/emmnet-angles-of_20260211_191017 \
  --split train \
  --sequence 0010 \
  --out analysis/outputs/flow/event_flow_pipeline_0010.png
```

## Dataset and Performance

- `dataset/plot_sequences.py` plots IMU, range-meter, trajectory, and timestep
  information for all train/test sequences under `outputs/dataset/sequences/`.
  Its model and output paths are module-level constants, and it executes at
  import time.
- `performance/benchmark_inference_latency.py` benchmarks warmed-up batch-1
  inference on CPU and/or CUDA. Results are printed to stdout and no files are
  written.

```bash
python analysis/performance/benchmark_inference_latency.py \
  --model-dir weights/emmnet-angles-of_20260209_144255 \
  --devices cpu cuda
```

## Publishing

`publishing/regenerate_manuscript_figures_large_fonts.sh` regenerates the
large-font journal, latent-comparison, manifold, and held-out revision figures
from existing latent artifacts, then collects the nine manuscript-ready PNGs in
`outputs/paper/manuscript_figures_large_fonts/`.

```bash
bash analysis/publishing/regenerate_manuscript_figures_large_fonts.sh
```

The helper defaults to the project Conda interpreter. Override `PYTHON` or
`FONT_SIZE` in the environment when needed.

## Output Policy

- Put new generated results in the matching `analysis/outputs/<theme>/`
  directory rather than beside source scripts.
- Treat checkpoint and extracted-array inputs as read-only.
- Do not commit `__pycache__/` or temporary cache directories.
- Reserve Markdown files for repository documentation such as this README;
  analysis scripts write numerical tables as CSV rather than Markdown.
