#!/usr/bin/env python3
"""Diagnose trajectory-wise calibration of the frozen revision ridge probes.

This fixed-configuration follow-up loads the with-flow and without-flow train and
validation packs from `analysis/outputs/revision/latents/`, fits velocity ridge
probes on the train split only, and measures component- and trajectory-level
calibration.

It writes CSV metrics and three PNG plots under
`analysis/outputs/revision/validation_analysis/probe_diagnostic/`. There is no
CLI; paths, model labels, and output names are constants below.
"""

from __future__ import annotations

import os
import sys
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.latent.compare_latent_spaces import (
    _ridge_predict,
    _ridge_regression,
    _zscore_apply,
    _zscore_fit,
    load_latent_pack,
)


LATENT_DIR = ROOT / "analysis" / "outputs" / "revision" / "latents"
ANALYSIS_DIR = ROOT / "analysis" / "outputs" / "revision" / "validation_analysis"
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    component, trajectory, cache = evaluate()
    component.to_csv(OUT_DIR / "per_component_metrics.csv", index=False, float_format="%.12g")
    trajectory.to_csv(OUT_DIR / "per_trajectory_metrics.csv", index=False, float_format="%.12g")
    save_figures(cache, trajectory)
    print(f"Saved frozen-probe diagnostic to {OUT_DIR}")


if __name__ == "__main__":
    main()
