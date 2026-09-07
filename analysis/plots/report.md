# Journal Latent Analysis Report

- Alignment mode: `index_prefix`
- Aligned samples: `4337`
- Curated paper figures: `figures_paper/`
- Curated interpretation notes: `figures_paper/paper_plot_interpretation.md`

## Key Performance Change

- `with_flow_best` RMSE: `9.269147`
- `without_flow_best` RMSE: `17.494839`
- Relative RMSE improvement: `47.02%`

## Navigation-Pipeline Use Of The Latent Space

### with_flow_best
- `pca_dim_95`: `3.0000`
- `ridge_z_to_velocity_r2`: `0.9645`
- `cca_top1_z_vs_velocity_speed`: `0.9993`
- `best_high_error_roc_auc`: `0.8380`
- `best_score_reject10_high_error_recall`: `0.3664`
- `best_score_reject10_retained_rmse`: `7.1858`
- `mean_latent_smoothness`: `2.3207`

### without_flow_best
- `pca_dim_95`: `4.0000`
- `ridge_z_to_velocity_r2`: `0.9599`
- `cca_top1_z_vs_velocity_speed`: `0.9991`
- `best_high_error_roc_auc`: `0.8032`
- `best_score_reject10_high_error_recall`: `0.2604`
- `best_score_reject10_retained_rmse`: `16.5660`
- `mean_latent_smoothness`: `2.7088`

Interpretation: these metrics assess whether the latent can serve as a compact navigation state, a lightweight readout interface, and a risk monitor for fallback/rejection policies. They do not turn the network into a formally calibrated estimator.

## Strongest Robustness Gains

- `low_event_density`: delta `-10.5533`, relative gain `45.58%`
- `all`: delta `-8.2257`, relative gain `47.02%`
- `high_lateral_speed`: delta `-7.4906`, relative gain `33.66%`
- `low_lateral_speed`: delta `-7.3698`, relative gain `77.04%`
- `low_vertical_speed`: delta `-5.8548`, relative gain `57.53%`

## Reject-option Reliability

- `with_flow_best` Mahalanobis reject 10%: RMSE `9.2691` -> `7.1858`
- `without_flow_best` Mahalanobis reject 10%: RMSE `17.4948` -> `16.9185`

## Operational Gate / Fallback Policy

- `with_flow_best` best 10% gate uses `mahalanobis`: high-error recall `0.366`, rejection precision `0.366`, retained RMSE `7.1858`.
- `without_flow_best` best 10% gate uses `event_density`: high-error recall `0.260`, rejection precision `0.260`, retained RMSE `16.5660`.
- Use this table to choose when a navigation stack should fall back to a classical estimator, increase uncertainty, or request conservative control.

## High-error Detection

- `with_flow_best` `mahalanobis`: ROC AUC `0.838`, PR AUC `0.407`, Spearman `0.686`
- `with_flow_best` `latent_norm`: ROC AUC `0.814`, PR AUC `0.263`, Spearman `0.642`
- `without_flow_best` `attention_range`: ROC AUC `0.803`, PR AUC `0.211`, Spearman `0.734`
- `without_flow_best` `latent_norm`: ROC AUC `0.728`, PR AUC `0.180`, Spearman `0.720`
- `without_flow_best` `attention_attitude`: ROC AUC `0.712`, PR AUC `0.185`, Spearman `0.305`
- `with_flow_best` `attention_event`: ROC AUC `0.670`, PR AUC `0.314`, Spearman `0.263`

## Attention Correlations

- `without_flow_best` attention `range` vs `speed`: Spearman `-0.760`
- `without_flow_best` attention `range` vs `error`: Spearman `-0.734`
- `with_flow_best` attention `event` vs `event_density`: Spearman `0.691`
- `with_flow_best` attention `attitude` vs `event_density`: Spearman `-0.618`
- `without_flow_best` attention `event` vs `speed`: Spearman `0.479`
- `with_flow_best` attention `omega` vs `event_density`: Spearman `0.479`
- `without_flow_best` attention `attitude` vs `event_density`: Spearman `-0.418`
- `with_flow_best` attention `attitude` vs `error`: Spearman `0.364`

## Latent Geometry And Physical Alignment

- `with_flow_best` latent-to-velocity ridge probe R2: `0.9645`
- `without_flow_best` latent-to-velocity ridge probe R2: `0.9599`
- PCA/UMAP figures are saved under `figures/`.

## Temporal Diagnostics

- Temporal-contiguous diagnostics were computed from sequence IDs and timestamps.

## Scientific Claim Checklist

1. Flow self-supervision improves final velocity RMSE, especially in hard regimes: see `basic_performance` and `robustness_slices`.
2. Structured latent space can be used pragmatically in navigation: see `navigation_readiness_summary` and `navigation_gate_policy`.
3. Flow supervision reshapes latent geometry: see PCA/UMAP and physical-alignment outputs.
4. Latent distances as reliability indicators: see `reject_option`, `high_error_detection`, and `navigation_gate_policy`.
5. Context-dependent modality reliance: see `attention_correlations` and attention-vs-context plots.
6. State-estimator-like behavior: see physical alignment and temporal diagnostics; avoid formal uncertainty claims.
