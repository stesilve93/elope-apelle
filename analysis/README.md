# Analysis Scripts

This folder contains exploratory and paper-support scripts. They are separate from
the main training, evaluation, and submission entry points in the repository root.

Run these commands from the repository root so the default relative paths resolve
to `elope_data/`, `weights/`, `plots/`, and `sequence_*` output folders.

## Latent Analysis

- `compare_latent_spaces.py`: compare two latent spaces from saved latent `.npz`
  files or from trained model folders.
- `visualize_latent_manifold.py`: create PCA, t-SNE, or UMAP visualizations from
  an extracted latent `.npz` file.
- `inspect_latents_angles.py`: inspect fused latent logs and attention summaries.
- `inspect_classical_hints.py`: compute classical-behavior probes from extracted
  latent packs.

## Event And Flow Inspection

- `inspect_events.py`: render event windows, optionally with EVFlowNet outputs.
- `inspect_flow_preds.py`: render model flow predictions and compare them against
  EVFlowNet outputs.
- `plot_sequences.py`: plot trajectory, IMU, and range-meter signals for dataset
  sequences.
- `testevflow.py`: scratch script for EVFlowNet checks.

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
```
