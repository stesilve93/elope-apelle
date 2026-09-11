#!/usr/bin/env bash
set -euo pipefail

# Regenerate the manuscript and held-out-revision figures from existing latent
# artifacts. This script does not run model inference or modify ELOPE inputs.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/staff/silvestrini/miniconda3/envs/elope/bin/python}"
FONT_SIZE="${FONT_SIZE:-15}"

# UMAP/Matplotlib need writable cache locations on the shared filesystem.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/elope-numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/elope-mpl}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

cd "$PROJECT_DIR"

# Original journal composites (Figures 2--4).
"$PYTHON" analysis/latent/analyze_journal_latents.py \
  --with-flow-dir plots/latent_compare_best_3 \
  --without-flow-dir plots/latent_compare_best_3 \
  --out-dir analysis/outputs/paper/plots_large_fonts \
  --font-size "$FONT_SIZE" \
  --no-figure-titles

# Original comparison plots, including robustness_slices_rmse and
# latent_pca_speed_compare. Existing extracted NPZs are reused.
"$PYTHON" analysis/latent/compare_latent_spaces.py \
  --flow-latents plots/latent_compare_best_3/extracted_with_flow_best.npz \
  --noflow-latents plots/latent_compare_best_3/extracted_without_flow_best.npz \
  --flow-name with_flow_best \
  --noflow-name without_flow_best \
  --out plots/latent_compare_best_3_large_fonts \
  --num-workers 0 \
  --font-size "$FONT_SIZE" \
  --no-figure-titles

# Original UMAP speed/error/event-density landscapes.
"$PYTHON" analysis/latent/visualize_latent_manifold.py \
  --npz plots/latent_compare_best_3/extracted_with_flow_best.npz \
  --name with_flow_best \
  --out plots/latent_compare_best_3_large_fonts/latent_manifold_flow_best_umap \
  --method umap \
  --font-size "$FONT_SIZE" \
  --no-figure-titles

# Held-out revision figures. The four existing split artifacts are reused.
"$PYTHON" analysis/latent/revision/recompute_revision_validation.py \
  --reuse-artifacts \
  --device cpu \
  --num-workers 0 \
  --font-size "$FONT_SIZE"

MANUSCRIPT_DIR="analysis/outputs/paper/manuscript_figures_large_fonts"
mkdir -p "$MANUSCRIPT_DIR"

# Collect the exact nine manuscript replacements in one directory. In
# particular, use the original eight-category comparison robustness chart,
# not the distinct journal-analysis chart that happens to share its basename.
cp -f analysis/outputs/paper/plots_large_fonts/figures_paper/paper_figure_2_navigation_gate.png \
  "$MANUSCRIPT_DIR/paper_figure_2_navigation_gate.png"
cp -f analysis/outputs/paper/plots_large_fonts/figures_paper/paper_figure_3_risk_calibration.png \
  "$MANUSCRIPT_DIR/paper_figure_3_risk_calibration.png"
cp -f analysis/outputs/paper/plots_large_fonts/figures_paper/paper_figure_4_latent_manifold.png \
  "$MANUSCRIPT_DIR/paper_figure_4_latent_manifold.png"
cp -f plots/latent_compare_best_3_large_fonts/plots/robustness_slices_rmse.png \
  "$MANUSCRIPT_DIR/robustness_slices_rmse.png"
cp -f plots/latent_compare_best_3_large_fonts/plots/latent_pca_speed_compare.png \
  "$MANUSCRIPT_DIR/latent_pca_speed_compare.png"
cp -f plots/latent_compare_best_3_large_fonts/latent_manifold_flow_best_umap/manifold_speed_landscape_hexbin.png \
  "$MANUSCRIPT_DIR/manifold_speed_landscape_hexbin_umap.png"
cp -f plots/latent_compare_best_3_large_fonts/latent_manifold_flow_best_umap/manifold_error_landscape_hexbin.png \
  "$MANUSCRIPT_DIR/manifold_error_landscape_hexbin_umap.png"
cp -f plots/latent_compare_best_3_large_fonts/latent_manifold_flow_best_umap/manifold_eventdensity_landscape_hexbin.png \
  "$MANUSCRIPT_DIR/manifold_eventdensity_landscape_hexbin_umap.png"
cp -f analysis/outputs/revision/validation_analysis/figures/validation_risk_calibration.png \
  "$MANUSCRIPT_DIR/validation_risk_calibration.png"

echo "Large-font figures regenerated with base font size ${FONT_SIZE}."
echo "Nine manuscript-ready PNGs: ${PROJECT_DIR}/${MANUSCRIPT_DIR}"
