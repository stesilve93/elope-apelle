# Curated Paper Figure Interpretation

These four figures are intended as the paper-facing subset. The remaining figures and tables are diagnostic support material.

## Figure 1 - Robustness Summary

**Critical evaluation.** This is the cleanest plot for demonstrating practical estimation gain. It compares velocity RMSE in regimes that matter to navigation: sparse events, high speed, high lateral speed, vertical motion, and high angular excitation when available.

**Interpretation.** The with-flow model reduces global RMSE from 17.49 to 9.27, a 47.0% relative improvement. The important point is not only the average gain, but whether the gain persists in hard regimes. If the sparse-event/high-speed bars remain lower for with-flow, the auxiliary task is improving motion evidence where event-based navigation is most fragile.

**Caveat.** Slices are quantile-defined on this evaluation set. They are operationally useful stress tests, but not universal flight-envelope thresholds.

## Figure 2 - Latent-Derived Navigation Gate

**Critical evaluation.** This figure answers whether the latent can drive a fallback policy. The left panel shows the accuracy retained after routing the riskiest windows away from the learned estimator. The right panel shows whether those rejected windows actually contain high-error cases.

**Interpretation.** At a 10% fallback rate, `mahalanobis` reduces retained RMSE to 7.19 and catches 36.6% of high-error windows. A useful navigation monitor should reduce retained RMSE without rejecting arbitrary samples. Mahalanobis distance is especially interpretable here because it measures how far a window lies from the learned latent distribution.

**Caveat.** This is not calibrated uncertainty. It is a ranking signal for fallback, covariance inflation, or conservative control.

## Figure 3 - Risk Calibration By Latent Distance

**Critical evaluation.** This plot checks monotonicity: as latent Mahalanobis distance increases, prediction error should rise. That is the key property needed for a practical risk monitor.

**Interpretation.** A rising curve means latent geometry contains reliability information, not just task information. The interquartile band is important: wide bands indicate that the score is useful statistically but not deterministic at the single-sample level.

**Caveat.** If high deciles flatten or become noisy, the latent score should be used only as one cue together with event density, IMU excitation, and classical residual checks.

## Figure 4 - Compact Latent Manifold With Failure Overlay

**Critical evaluation.** This is the best visual explanation of the latent as a navigation state. It now compares the same PCA manifold colored by physical speed and by normalized within-trajectory progress, so it directly checks whether the speed structure is merely a trajectory-phase artifact.

**Interpretation.** The with-flow latent is compact: `pca_dim_95=3` and participation ratio `2.52` in the summary table. A coherent speed gradient supports using the latent as a compact state-like representation. In this run, speed is strongly organized along PC2 (Spearman 0.91), whereas trajectory progress is weakly related to PC2 (Spearman -0.10) and speed is only weakly related to progress overall (Spearman -0.09). Thus the visible speed gradient should not be described as only an encoding of sample order. Red high-error rings reveal where the learned state is less reliable.

**Caveat.** PCA is a 2D projection and speed is itself physically correlated with landing phase in some trajectories. Use the side-by-side progress panel and `paper_figure_4_phase_correlations.csv` to avoid over-claiming that the network has learned speed independently of trajectory phase.
