# Analysis Scripts

This folder contains exploratory and paper-support scripts. They are separate from
the main training, evaluation, and submission entry points in the repository root.

Run these commands from the repository root so the default relative paths resolve
to `elope_data/`, `weights/`, `plots/`, and `sequence_*` output folders.

## Current Scripts

### Latent Analysis

- `compare_latent_spaces.py`: compare two latent spaces from saved latent `.npz`
  files or from trained model folders.
- `visualize_latent_manifold.py`: create PCA, t-SNE, or UMAP visualizations from
  an extracted latent `.npz` file.
- `inspect_latents_angles.py`: inspect fused latent logs and attention summaries.
- `inspect_classical_hints.py`: compute classical-behavior probes from extracted
  latent packs.
- `analyze_journal_latents.py`: generate publication tables and figures for
  performance, robustness slices, reject-option reliability, operational
  navigation gates/fallback policies, high-error detection, attention/context
  correlations, physical alignment, temporal diagnostics, and PCA/UMAP manifold
  analysis. It is intended to assess both the flow auxiliary-task effect and
  whether the learned latent is useful as a compact navigation-state and risk
  monitoring signal.

### Revision Validation

- `recompute_revision_validation.py`: re-extract or reuse fixed train/validation
  latent packs and generate leakage-free held-out revision metrics and figures.
- `diagnose_revision_probes.py`: inspect trajectory- and component-level
  calibration of the frozen revision ridge probes.

### Event And Flow Inspection

- `inspect_events.py`: render event windows, optionally with EVFlowNet outputs.
- `inspect_flow_preds.py`: render model flow predictions and compare them against
  EVFlowNet outputs.
- `visualize_event_flow_pipeline.py`: create a publication-style single-frame
  visualization of the causal event window, tensorized polarity channels, learned
  optical-flow head output, sparse flow vectors, and predicted velocity context.
- `testevflow.py`: scratch script for EVFlowNet checks.

### Dataset And Performance Inspection

- `plot_sequences.py`: plot trajectory, IMU, range-meter, and timestep data for
  every train/test sequence.
- `benchmark_inference_latency.py`: benchmark warmed-up batch-1 model inference
  on CPU and/or CUDA and print publication-ready results.

## Proposed Organization

The following layout separates source scripts from generated artifacts while
keeping related workflows together. It is a proposal only; the files have not
been moved because current sibling imports and hard-coded paths should be cleaned
up as part of the migration.

```text
analysis/
├── README.md
├── latent/
│   ├── compare_latent_spaces.py
│   ├── visualize_latent_manifold.py
│   ├── inspect_latents_angles.py
│   ├── inspect_classical_hints.py
│   └── analyze_journal_latents.py
├── latent/revision/
│   ├── recompute_revision_validation.py
│   └── diagnose_revision_probes.py
├── flow_visualization/
│   ├── inspect_events.py
│   ├── inspect_flow_preds.py
│   ├── visualize_event_flow_pipeline.py
│   └── test_evflownet.py
├── dataset/
│   └── plot_sequences.py
├── performance/
│   └── benchmark_inference_latency.py
├── publishing/
│   └── regenerate_manuscript_figures_large_fonts.sh
└── outputs/                 # generated; ideally gitignored
    ├── latent/
    ├── revision/
    ├── flow/
    └── paper/
```

Before moving files, convert cross-script imports such as
`from compare_latent_spaces import ...` to package imports, replace fixed paths
with CLI arguments, and decide which existing generated artifacts must remain
versioned. `__pycache__/` should be ignored rather than organized.

Examples:

```bash
python analysis/compare_latent_spaces.py \
  --flow-model-dir weights/emmnet-angles-of_YYYYMMDD_HHMMSS \
  --noflow-model-dir weights/emmnet-angles_YYYYMMDD_HHMMSS \
  --out plots/latent_compare

python analysis/visualize_latent_manifold.py \
  --npz plots/latent_compare/extracted_with_flow.npz \
  --out plots/latent_manifold \
  --interactive

python analysis/analyze_journal_latents.py \
  --with-flow-dir plots/latent_compare_best_3 \
  --without-flow-dir plots/latent_compare_best_3 \
  --out-dir plots/journal_latent_analysis

python analysis/visualize_event_flow_pipeline.py \
  --model-dir weights/emmnet-angles-of_20260211_191017 \
  --split train \
  --sequence 0010 \
  --out analysis/plots/figures_paper/event_flow_pipeline.png
```
