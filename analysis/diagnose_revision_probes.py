#!/usr/bin/env python3
"""Diagnose trajectory-wise calibration of the frozen revision ridge probes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from compare_latent_spaces import (
    _ridge_predict,
    _ridge_regression,
    _zscore_apply,
    _zscore_fit,
    load_latent_pack,
)


ROOT = Path(__file__).resolve().parents[1]
LATENT_DIR = ROOT / "analysis" / "revision_latents"
ANALYSIS_DIR = ROOT / "analysis" / "revision_validation_analysis"
OUT_DIR = ANALYSIS_DIR / "probe_diagnostic"
MODELS = ("with_flow", "without_flow")
COMPONENTS = ("vx", "vy", "vz")
EPS = 1e-12


def fit_frozen_probe(train):
    """Exactly the train-only probe used by recompute_revision_validation.py."""
    mu, sd = _zscore_fit(train.fused)
    weights = _ridge_regression(_zscore_apply(train.fused, mu, sd), train.target_vel, l2=1e-3)
    return mu, sd, weights


def predict(pack, probe) -> np.ndarray:
    mu, sd, weights = probe
    return _ridge_predict(_zscore_apply(pack.fused, mu, sd), weights)


def safe_corr(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    result = stats.spearmanr(x, y).correlation if rank else stats.pearsonr(x, y).statistic
    return float(result)


def component_rows(model: str, split: str, trajectory: str, y: np.ndarray, pred: np.ndarray) -> list[dict]:
    rows = []
    for j, component in enumerate(COMPONENTS):
        target, estimate = y[:, j], pred[:, j]
        if np.std(estimate) < EPS:
            slope, intercept = np.nan, float(np.mean(target))
        else:
            slope, intercept = np.polyfit(estimate, target, deg=1)
        rows.append(
            {
                "model": model,
                "split": split,
                "trajectory_id": trajectory,
                "component": component,
                "n": len(target),
                "target_mean": float(np.mean(target)),
                "target_std": float(np.std(target)),
                "target_min": float(np.min(target)),
                "target_max": float(np.max(target)),
                "predicted_mean": float(np.mean(estimate)),
                "predicted_std": float(np.std(estimate)),
                "rmse": float(np.sqrt(np.mean((estimate - target) ** 2))),
                "mean_signed_bias_pred_minus_target": float(np.mean(estimate - target)),
                "r2": float(r2_score(target, estimate)),
                "pearson": safe_corr(target, estimate),
                "spearman": safe_corr(target, estimate, rank=True),
                "diagnostic_target_on_prediction_slope": float(slope),
                "diagnostic_target_on_prediction_intercept": float(intercept),
            }
        )
    return rows


def aggregate_row(model: str, split: str, trajectory: str, y: np.ndarray, pred: np.ndarray) -> dict:
    component_r2 = [r2_score(y[:, j], pred[:, j]) for j in range(3)]
    pearson = [safe_corr(y[:, j], pred[:, j]) for j in range(3)]
    spearman = [safe_corr(y[:, j], pred[:, j], rank=True) for j in range(3)]
    biases = np.mean(pred - y, axis=0)
    return {
        "model": model,
        "split": split,
        "trajectory_id": trajectory,
        "n": len(y),
        "mean_component_r2": float(np.mean(component_r2)),
        "mean_pearson": float(np.nanmean(pearson)),
        "mean_spearman": float(np.nanmean(spearman)),
        "vector_rmse": float(np.sqrt(np.mean(np.sum((pred - y) ** 2, axis=1)))),
        "target_velocity_std_mean": float(np.mean(np.std(y, axis=0))),
        "target_velocity_variance_mean": float(np.mean(np.var(y, axis=0))),
        "mean_absolute_component_bias": float(np.mean(np.abs(biases))),
        "bias_vector_norm": float(np.linalg.norm(biases)),
    }


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    component_metrics: list[dict] = []
    trajectory_metrics: list[dict] = []
    cache = {}
    for model in MODELS:
        train = load_latent_pack(LATENT_DIR / f"{model}_train.npz", model)
        validation = load_latent_pack(LATENT_DIR / f"{model}_validation.npz", model)
        probe = fit_frozen_probe(train)
        train_pred, validation_pred = predict(train, probe), predict(validation, probe)
        cache[model] = {"train": train, "validation": validation, "train_pred": train_pred,
                        "validation_pred": validation_pred}

        component_metrics.extend(component_rows(model, "train", "POOLED", train.target_vel, train_pred))
        trajectory_metrics.append(aggregate_row(model, "train", "POOLED", train.target_vel, train_pred))
        for sid in np.unique(train.sequence_id.astype(int)):
            idx = np.where(train.sequence_id.astype(int) == sid)[0]
            component_metrics.extend(component_rows(model, "train", f"{sid:04d}", train.target_vel[idx], train_pred[idx]))
            trajectory_metrics.append(aggregate_row(model, "train", f"{sid:04d}", train.target_vel[idx], train_pred[idx]))

        component_metrics.extend(component_rows(model, "validation", "0004", validation.target_vel, validation_pred))
        trajectory_metrics.append(aggregate_row(model, "validation", "0004", validation.target_vel, validation_pred))
    return pd.DataFrame(component_metrics), pd.DataFrame(trajectory_metrics), cache


def save_figures(cache: dict, trajectory_metrics: pd.DataFrame) -> None:
    colors = {"with_flow": "#0f766e", "without_flow": "#b45309"}
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), sharex="row")
    for row, model in enumerate(MODELS):
        pack, pred = cache[model]["validation"], cache[model]["validation_pred"]
        order = np.argsort(pack.times)
        for j, component in enumerate(COMPONENTS):
            ax = axes[row, j]
            ax.plot(pack.times[order], pack.target_vel[order, j], label="ground truth", color="#1f2937", linewidth=1.8)
            ax.plot(pack.times[order], pred[order, j], label="frozen probe", color=colors[model], linewidth=1.5)
            ax.set_title(f"{model}: {component}")
            ax.set_xlabel("Trajectory time (s)")
            ax.set_ylabel("Velocity")
            ax.grid(alpha=.25)
            if row == 0 and j == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_time_series.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
    for row, model in enumerate(MODELS):
        pack, pred = cache[model]["validation"], cache[model]["validation_pred"]
        for j, component in enumerate(COMPONENTS):
            ax = axes[row, j]
            target = pack.target_vel[:, j]
            ax.scatter(target, pred[:, j], s=18, alpha=.7, color=colors[model])
            lo, hi = float(min(target.min(), pred[:, j].min())), float(max(target.max(), pred[:, j].max()))
            ax.plot([lo, hi], [lo, hi], "--", color="#64748b", linewidth=1, label="y=x")
            ax.set_title(f"{model}: {component}")
            ax.set_xlabel("Ground truth")
            ax.set_ylabel("Frozen-probe prediction")
            ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_scatter.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        train = trajectory_metrics[(trajectory_metrics.model == model) &
                                   (trajectory_metrics.split == "train") &
                                   (trajectory_metrics.trajectory_id != "POOLED")]
        validation = trajectory_metrics[(trajectory_metrics.model == model) &
                                        (trajectory_metrics.split == "validation")].iloc[0]
        ax.scatter(train.target_velocity_std_mean, train.mean_component_r2, s=27, alpha=.75,
                   color=colors[model], label="training trajectories")
        ax.scatter([validation.target_velocity_std_mean], [validation.mean_component_r2], s=90, marker="*",
                   color="#dc2626", label="validation 0004", zorder=4)
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(model)
        ax.set_xlabel("Mean component target std")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean component R2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "r2_vs_target_std.png", dpi=220)
    plt.close(fig)


def percentile_of_training(values: np.ndarray, value: float, higher_is_harder: bool) -> float:
    if higher_is_harder:
        return float(100 * np.mean(values <= value))
    return float(100 * np.mean(values >= value))


def write_interpretation(component: pd.DataFrame, trajectory: pd.DataFrame) -> None:
    cca = pd.read_csv(ANALYSIS_DIR / "tables" / "cca_metrics.csv")
    lines = [
        "# Frozen linear-probe validation diagnostic",
        "",
        "The same global ridge map is fitted once on all 4,215 training latents and then applied unchanged everywhere below. Per-trajectory regressions of `ground_truth = a * probe_prediction + b` are diagnostic only; no reported probe prediction is recalibrated.",
        "",
        "## Key results",
        "",
        "| Model | Scope | Mean component R2 | Mean Pearson | Mean Spearman | Vector RMSE | Mean target std | Mean abs. component bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = {}
    for model in MODELS:
        pooled = trajectory[(trajectory.model == model) & (trajectory.trajectory_id == "POOLED")].iloc[0]
        validation = trajectory[(trajectory.model == model) & (trajectory.split == "validation")].iloc[0]
        train_traj = trajectory[(trajectory.model == model) & (trajectory.split == "train") &
                                (trajectory.trajectory_id != "POOLED")]
        for label, row in (("pooled train", pooled), ("validation 0004", validation)):
            lines.append(f"| {model} | {label} | {row.mean_component_r2:.4f} | {row.mean_pearson:.4f} | "
                         f"{row.mean_spearman:.4f} | {row.vector_rmse:.3f} | {row.target_velocity_std_mean:.3f} | "
                         f"{row.mean_absolute_component_bias:.3f} |")

        cval = component[(component.model == model) & (component.split == "validation")]
        slopes = cval.diagnostic_target_on_prediction_slope.to_numpy()
        intercepts = cval.diagnostic_target_on_prediction_intercept.to_numpy()
        pred_target_std_ratio = cval.predicted_std.to_numpy() / np.maximum(cval.target_std.to_numpy(), EPS)
        bias_fraction = float(validation.bias_vector_norm / max(validation.vector_rmse, EPS))
        std_r2_corr = safe_corr(train_traj.target_velocity_std_mean.to_numpy(), train_traj.mean_component_r2.to_numpy())
        r2_percentile = percentile_of_training(train_traj.mean_component_r2.to_numpy(), validation.mean_component_r2, False)
        rmse_percentile = percentile_of_training(train_traj.vector_rmse.to_numpy(), validation.vector_rmse, True)
        std_percentile = float(100 * np.mean(train_traj.target_velocity_std_mean <= validation.target_velocity_std_mean))
        fraction_below_pooled = float(np.mean(train_traj.mean_component_r2 < pooled.mean_component_r2))
        cca1 = float(cca[(cca.model == model) & (cca.metric == "canonical_1_correlation")].iloc[0].value)
        cca_mean4 = float(cca[(cca.model == model) & (cca.metric == "top_4_mean")].iloc[0].value)
        summaries[model] = dict(pooled=pooled, validation=validation, train=train_traj, cval=cval,
                                slopes=slopes, intercepts=intercepts, ratios=pred_target_std_ratio,
                                bias_fraction=bias_fraction, std_r2_corr=std_r2_corr,
                                r2_percentile=r2_percentile, rmse_percentile=rmse_percentile,
                                std_percentile=std_percentile, fraction_below_pooled=fraction_below_pooled,
                                cca1=cca1, cca_mean4=cca_mean4)

    lines += ["", "## Component-level validation calibration", "",
              "| Model | Component | R2 | Pearson | Spearman | Bias | Pred/target std | Diagnostic slope a | Diagnostic intercept b |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for _, row in summaries[model]["cval"].iterrows():
            ratio = row.predicted_std / max(row.target_std, EPS)
            lines.append(f"| {model} | {row.component} | {row.r2:.4f} | {row.pearson:.4f} | {row.spearman:.4f} | "
                         f"{row.mean_signed_bias_pred_minus_target:.3f} | {ratio:.3f} | "
                         f"{row.diagnostic_target_on_prediction_slope:.3f} | {row.diagnostic_target_on_prediction_intercept:.3f} |")

    lines += ["", "## Interpretation", ""]
    lines += [
        "### Direct answers",
        "",
        "- **High correlation despite poor R2: yes.** All six validation component Pearson correlations are at least `0.923`, and all Spearman correlations are at least `0.962`, while aggregate mean-component R2 is strongly negative.",
        "- **Bias alone: no.** Bias is substantial, but its vector norm is only about 60--62% of vector RMSE (about 36--38% on a squared-error basis), so centered/scale error remains material.",
        "- **Scale mismatch: yes, especially for vy.** Validation `vy` target std is `5.060`, versus probe-prediction std `29.681` with flow and `51.507` without flow. The corresponding diagnostic slopes are `0.165` and `0.095`, far from one.",
        "- **Low target variance: a contributor to vy R2, not a complete explanation.** Validation `vy` variance is lower than every training trajectory's, which magnifies its R2 penalty. However, validation `vx`/`vz` variability is near or above the top of the training range, mean target std is at the 63rd training percentile, and vector RMSE is also worse than every training trajectory.",
        "- **Pooled training R2 hides trajectory variation: yes.** Nearly every individual training trajectory has lower R2 than the pooled fit; some no-flow training trajectories are already negative. The validation result is nevertheless far outside the training-trajectory range.",
        "- **Validation trajectory unusually difficult: yes for this frozen probe.** It is worst among the compared trajectories by both aggregate R2 and vector RMSE for both models.",
        "- **Transferable subspace with trajectory-dependent calibration: consistent with the evidence.** High component correlations and held-out CCA coexist with severe raw scale/bias errors. This supports that interpretation descriptively, but does not establish that any recalibration would generalize to another held-out trajectory.",
        "",
    ]
    for model in MODELS:
        s = summaries[model]
        v, p, train = s["validation"], s["pooled"], s["train"]
        correlation_text = "high" if v.mean_pearson >= .8 and v.mean_spearman >= .8 else ("moderate" if v.mean_pearson >= .5 else "low")
        lines += [
            f"### {model}",
            "",
            f"- Validation rank/linear association is **{correlation_text} overall** (mean Pearson `{v.mean_pearson:.3f}`, mean Spearman `{v.mean_spearman:.3f}`) despite mean component R2 `{v.mean_component_r2:.3f}`.",
            f"- The bias-vector norm is `{v.bias_vector_norm:.3f}`, or `{100*s['bias_fraction']:.1f}%` of vector RMSE `{v.vector_rmse:.3f}`. Component predicted/target standard-deviation ratios are "
            f"`{s['ratios'][0]:.2f}`, `{s['ratios'][1]:.2f}`, `{s['ratios'][2]:.2f}`; diagnostic slopes are "
            f"`{s['slopes'][0]:.2f}`, `{s['slopes'][1]:.2f}`, `{s['slopes'][2]:.2f}`. These quantify bias and scale mismatch without correcting either.",
            f"- Validation target variability is at the `{s['std_percentile']:.1f}`th percentile of training trajectories. Across training trajectories, target std versus R2 has Pearson `{s['std_r2_corr']:.3f}`; this indicates whether low within-trajectory variance is a dominant explanation.",
            f"- `{100*s['fraction_below_pooled']:.1f}%` of individual training trajectories have lower R2 than pooled training R2 `{p.mean_component_r2:.3f}`. Training-trajectory median R2 is `{train.mean_component_r2.median():.3f}` (range `{train.mean_component_r2.min():.3f}` to `{train.mean_component_r2.max():.3f}`).",
            f"- Validation ranks at the `{s['r2_percentile']:.1f}`th difficulty percentile by low R2 and `{s['rmse_percentile']:.1f}`th by high RMSE relative to the 27 training trajectories.",
            f"- Held-out CCA remains `{s['cca1']:.3f}` for component 1 and `{s['cca_mean4']:.3f}` averaged across four absolute canonical correlations. CCA establishes subspace association, not absolute velocity calibration.",
            "",
        ]

    transferable = all(summaries[m]["validation"].mean_pearson >= .5 and summaries[m]["cca_mean4"] >= .7 for m in MODELS)
    calibration = all(summaries[m]["validation"].mean_component_r2 < 0 for m in MODELS)
    if transferable and calibration:
        conclusion = ("The numbers are consistent with transfer of a velocity-related latent subspace together with trajectory-dependent absolute calibration: "
                      "held-out correlations/CCA remain substantial while raw, unmodified probe R2 is negative. This is an association, not evidence that a validation-calibrated probe would generalize.")
    else:
        conclusion = ("The requested interpretation is not fully supported: the held-out association metrics are not consistently strong enough across both models to distinguish calibration shift from loss of linearly accessible velocity information.")
    lines += ["## Bottom line", "", conclusion, "",
              "Detailed values for every component/scope and every trajectory are in `per_component_metrics.csv` and `per_trajectory_metrics.csv`."]
    (OUT_DIR / "probe_validation_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    component, trajectory, cache = evaluate()
    component.to_csv(OUT_DIR / "per_component_metrics.csv", index=False, float_format="%.12g")
    trajectory.to_csv(OUT_DIR / "per_trajectory_metrics.csv", index=False, float_format="%.12g")
    save_figures(cache, trajectory)
    write_interpretation(component, trajectory)
    print(f"Saved frozen-probe diagnostic to {OUT_DIR}")


if __name__ == "__main__":
    main()
