#!/usr/bin/env python3
"""Generate journal-ready latent, reliability, attention, and robustness results.

Purpose
-------
Compare with-flow and without-flow model outputs, quantify prediction quality and
latent-space structure, and evaluate whether latent/attention signals are useful
for error detection, rejection, and navigation gating.

Expected input format
---------------------
Pass one directory for the with-flow model and one directory for the without-flow
model. Each directory should contain one or more `.npz`, `.npy`, `.csv`, or
optional PyTorch `.pt/.pth` files with arrays using any of these common names:

- latents: `fused`, `z`, `Z`, `latent`, `latents`, `embedding`, `embeddings`
- predictions: `pred`, `prediction`, `predictions`, `y_pred`, `vel_pred`
- ground truth velocity: `target_vel`, `y_true`, `target`, `targets`,
  `vel_true`, `velocity_true`
- metadata: `keys`, `sample_keys`, `times`, `timestamps`, `sequence_id`,
  `seq_id`, `event_density`, `omega`, `angular_velocity`
- attention: `attention`, `attn`, `attention_weights`

The repository's existing extracted latent files are supported directly, e.g.
`extracted_with_flow_best.npz` with keys `fused`, `pred`, `target_vel`, `times`,
`event_density`, `attention`, and `sequence_id`.

Outputs
-------
`--out-dir` is populated with `report.md`, CSV/Markdown tables, PNG diagnostic
and paper figures, cached PCA/UMAP and reliability arrays, and run metadata.
The script does not modify model checkpoints or source latent files.

Example
-------
python analysis/analyze_journal_latents.py \\
  --with-flow-dir plots/latent_compare_best_3 \\
  --without-flow-dir plots/latent_compare_best_3 \\
  --out-dir plots/journal_latent_analysis \\
  --make-umap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import warnings

SHOW_FIGURE_TITLES = True
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TOKEN_GROUPS = {
    "event": [0, 1, 2, 3],
    "omega": [4],
    "range": [5],
    "attitude": [6],
}

ALIASES = {
    "z": {"fused", "z", "Z", "latent", "latents", "embedding", "embeddings"},
    "pred": {"pred", "preds", "prediction", "predictions", "y_pred", "vel_pred", "velocity_pred"},
    "true": {"target_vel", "y_true", "target", "targets", "vel_true", "velocity_true", "gt", "ground_truth"},
    "keys": {"key", "keys", "sample_key", "sample_keys", "id", "ids"},
    "times": {"time", "times", "timestamp", "timestamps", "t"},
    "seq": {"sequence_id", "sequence_ids", "seq_id", "seq_ids", "sequence", "seq"},
    "event_density": {"event_density", "density", "event_count", "event_counts"},
    "omega": {"omega", "angular_velocity", "angular_vel", "angular_rate", "imu_omega", "w"},
    "attention": {"attention", "attn", "attention_weights", "attention_matrix"},
    "speed": {"speed", "velocity_norm", "vel_norm"},
}


@dataclass
class ModelOutputs:
    name: str
    z: np.ndarray
    pred: np.ndarray
    true: np.ndarray
    keys: np.ndarray | None = None
    times: np.ndarray | None = None
    seq: np.ndarray | None = None
    event_density: np.ndarray | None = None
    omega: np.ndarray | None = None
    attention: np.ndarray | None = None
    speed: np.ndarray | None = None
    source_files: tuple[str, ...] = ()

    def subset(self, idx: np.ndarray) -> "ModelOutputs":
        idx = np.asarray(idx, dtype=int)

        def take(x: np.ndarray | None) -> np.ndarray | None:
            return x[idx] if x is not None and len(x) >= len(self.z) else x

        return ModelOutputs(
            name=self.name,
            z=self.z[idx],
            pred=self.pred[idx],
            true=self.true[idx],
            keys=take(self.keys),
            times=take(self.times),
            seq=take(self.seq),
            event_density=take(self.event_density),
            omega=take(self.omega),
            attention=take(self.attention),
            speed=take(self.speed),
            source_files=self.source_files,
        )


def setup_style(font_size: float = 10) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": max(font_size - 2, 1),
            "ytick.labelsize": max(font_size - 2, 1),
            "legend.fontsize": max(font_size - 2, 1),
            "legend.title_fontsize": max(font_size - 1, 1),
            "legend.frameon": False,
        }
    )


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "figures": out_dir / "figures",
        "figures_paper": out_dir / "figures_paper",
        "tables": out_dir / "tables",
        "arrays": out_dir / "arrays",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def canonical_name(name: str) -> str | None:
    clean = name.strip().lower()
    for key, aliases in ALIASES.items():
        if clean in {a.lower() for a in aliases}:
            return key
    return None


def flatten_mapping(obj: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            key_s = str(key)
            out[prefix + key_s] = val
            if isinstance(val, dict):
                out.update(flatten_mapping(val, prefix=prefix + key_s + "."))
    return out


def to_numpy(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif hasattr(x, "cpu") and hasattr(x, "numpy"):
            x = x.cpu().numpy()
        return np.asarray(x)
    except Exception:
        return None


def file_preference_score(path: Path, model_name: str = "") -> int:
    s = path.name.lower()
    model = model_name.lower()
    score = 0
    if "extracted" in s:
        score += 4
    if "latent" in s:
        score += 3
    if "best" in s:
        score += 2
    if path.suffix.lower() == ".npz":
        score += 1
    wants_without = "without" in model or "noflow" in model or "no_flow" in model
    wants_with = "with" in model and not wants_without
    is_without = "without" in s or "noflow" in s or "no_flow" in s or "no-flow" in s
    is_with = "with_flow" in s or "with-flow" in s or "flow_best" in s
    if wants_without:
        score += 12 if is_without else 0
        score -= 8 if is_with and not is_without else 0
    elif wants_with:
        score += 12 if is_with and not is_without else 0
        score -= 8 if is_without else 0
    return score


def discover_files(path: Path, model_name: str = "") -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".npz", ".npy", ".csv", ".pt", ".pth"}
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    preferred = [(file_preference_score(p, model_name), p) for p in files]
    preferred.sort(key=lambda x: (-x[0], str(x[1])))
    return [p for _, p in preferred]


def collect_from_npz(path: Path) -> dict[str, list[np.ndarray]]:
    found: dict[str, list[np.ndarray]] = {}
    with np.load(path, allow_pickle=True) as npz:
        for key in npz.files:
            canon = canonical_name(key)
            if canon is None:
                continue
            arr = npz[key]
            if arr.dtype == object and arr.shape == () and arr.item() is None:
                continue
            found.setdefault(canon, []).append(np.asarray(arr))
    return found


def collect_from_npy(path: Path) -> dict[str, list[np.ndarray]]:
    canon = canonical_name(path.stem)
    if canon is None:
        return {}
    return {canon: [np.load(path, allow_pickle=True)]}


def collect_from_csv(path: Path) -> dict[str, list[np.ndarray]]:
    df = pd.read_csv(path)
    found: dict[str, list[np.ndarray]] = {}
    lower_cols = {c.lower(): c for c in df.columns}

    triples = {
        "pred": ["pred_vx", "pred_vy", "pred_vz"],
        "true": ["vx", "vy", "vz"],
        "omega": ["omega_x", "omega_y", "omega_z"],
    }
    for canon, cols in triples.items():
        if all(c in lower_cols for c in cols):
            found.setdefault(canon, []).append(df[[lower_cols[c] for c in cols]].to_numpy())

    for col in df.columns:
        canon = canonical_name(col)
        if canon is None:
            continue
        vals = df[col].to_numpy()
        found.setdefault(canon, []).append(vals)
    return found


def collect_from_torch(path: Path) -> dict[str, list[np.ndarray]]:
    try:
        import torch
    except Exception as exc:
        warnings.warn(f"Skipping {path}: PyTorch is needed for .pt/.pth files ({exc}).")
        return {}

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    flat = flatten_mapping(obj) if isinstance(obj, dict) else {}
    found: dict[str, list[np.ndarray]] = {}
    for key, val in flat.items():
        canon = canonical_name(key.split(".")[-1])
        if canon is None:
            continue
        arr = to_numpy(val)
        if arr is not None:
            found.setdefault(canon, []).append(arr)
    return found


def merge_collections(files: list[Path]) -> dict[str, list[np.ndarray]]:
    found: dict[str, list[np.ndarray]] = {}
    readers = {
        ".npz": collect_from_npz,
        ".npy": collect_from_npy,
        ".csv": collect_from_csv,
        ".pt": collect_from_torch,
        ".pth": collect_from_torch,
    }
    for path in files:
        reader = readers.get(path.suffix.lower())
        if reader is None:
            continue
        try:
            part = reader(path)
        except Exception as exc:
            warnings.warn(f"Could not read {path}: {exc}")
            continue
        for key, arrays in part.items():
            found.setdefault(key, []).extend(arrays)
    return found


def read_one_file(path: Path) -> dict[str, list[np.ndarray]]:
    readers = {
        ".npz": collect_from_npz,
        ".npy": collect_from_npy,
        ".csv": collect_from_csv,
        ".pt": collect_from_torch,
        ".pth": collect_from_torch,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        return {}
    return reader(path)


def choose_array(name: str, arrays: list[np.ndarray]) -> np.ndarray | None:
    arrays = [np.asarray(a) for a in arrays if a is not None and np.asarray(a).size > 0]
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]

    compatible = []
    for arr in arrays:
        if arr.ndim == 0:
            continue
        compatible.append(arr)
    if not compatible:
        return arrays[0]

    first_shape = compatible[0].shape[1:]
    if all(a.shape[1:] == first_shape for a in compatible):
        return np.concatenate(compatible, axis=0)

    warnings.warn(
        f"Multiple incompatible arrays found for `{name}`. Using the longest first-axis array."
    )
    return max(compatible, key=lambda x: x.shape[0])


def normalize_vectors(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1 and name in {"event_density", "speed", "times", "seq", "keys"}:
        return arr.reshape(-1)
    if arr.ndim > 2 and name in {"z", "pred", "true", "omega"}:
        arr = arr.reshape(arr.shape[0], -1)
    if name == "true" and arr.shape[1] >= 6:
        return arr[:, 3:6].astype(float)
    if name == "omega" and arr.ndim == 2 and arr.shape[1] >= 6:
        return arr[:, 3:6].astype(float)
    if name in {"pred", "true", "omega"} and arr.ndim == 2 and arr.shape[1] > 3:
        return arr[:, :3].astype(float)
    if name == "z":
        return arr.astype(float)
    return arr


def load_outputs(path: Path, name: str) -> ModelOutputs:
    files = discover_files(path, name)
    if not files:
        raise FileNotFoundError(f"No supported input files found under {path}")

    found = {}
    source_files: list[Path] = []
    if path.is_dir():
        for file_path in files:
            try:
                one = read_one_file(file_path)
            except Exception:
                continue
            if all(key in one for key in ("z", "pred", "true")) and file_preference_score(file_path, name) >= 10:
                found = one
                source_files = [file_path]
                break
    if not found:
        found = merge_collections(files)
        source_files = files
    selected = {key: choose_array(key, arrays) for key, arrays in found.items()}
    selected = {key: normalize_vectors(arr, key) for key, arr in selected.items() if arr is not None}

    missing = [key for key in ("z", "pred", "true") if key not in selected]
    if missing:
        keys = ", ".join(sorted(found.keys())) or "none"
        raise ValueError(
            f"{name}: missing required arrays {missing} in {path}. "
            f"Recognized array groups: {keys}."
        )

    n = min(len(selected["z"]), len(selected["pred"]), len(selected["true"]))
    if n == 0:
        raise ValueError(f"{name}: required arrays are empty.")

    def trim(key: str) -> np.ndarray | None:
        arr = selected.get(key)
        if arr is None:
            return None
        if len(arr) < n:
            warnings.warn(f"{name}: optional `{key}` has length {len(arr)} < {n}; skipping it.")
            return None
        return arr[:n]

    return ModelOutputs(
        name=name,
        z=selected["z"][:n],
        pred=selected["pred"][:n],
        true=selected["true"][:n],
        keys=trim("keys"),
        times=trim("times"),
        seq=trim("seq"),
        event_density=trim("event_density"),
        omega=trim("omega"),
        attention=trim("attention"),
        speed=trim("speed"),
        source_files=tuple(str(p) for p in source_files),
    )


def stable_key_array(model: ModelOutputs) -> np.ndarray | None:
    if model.keys is not None:
        return np.asarray([str(x) for x in model.keys.reshape(-1)])
    if model.seq is not None and model.times is not None:
        seq = model.seq.reshape(-1)
        times = model.times.reshape(-1)
        return np.asarray([f"{seq[i]}::{float(times[i]):.9f}" for i in range(len(seq))])
    return None


def align_models(a: ModelOutputs, b: ModelOutputs) -> tuple[ModelOutputs, ModelOutputs, str]:
    ka = stable_key_array(a)
    kb = stable_key_array(b)
    if ka is not None and kb is not None:
        unique_a = len(np.unique(ka)) == len(ka)
        unique_b = len(np.unique(kb)) == len(kb)
        if unique_a and unique_b:
            pos_b = {k: i for i, k in enumerate(kb)}
            idx_a, idx_b = [], []
            for i, key in enumerate(ka):
                if key in pos_b:
                    idx_a.append(i)
                    idx_b.append(pos_b[key])
            if idx_a:
                return a.subset(np.asarray(idx_a)), b.subset(np.asarray(idx_b)), "key_match"
            warnings.warn("No common keys found; falling back to index alignment.")
        else:
            warnings.warn(
                "Alignment keys are not unique; falling back to index alignment. "
                "For temporal diagnostics, sequence IDs and timestamps are still used within each model."
            )

    n = min(len(a.z), len(b.z))
    return a.subset(np.arange(n)), b.subset(np.arange(n)), "index_prefix"


def velocity_errors(model: ModelOutputs) -> np.ndarray:
    return np.linalg.norm(model.pred - model.true, axis=1)


def speed_values(model: ModelOutputs) -> np.ndarray:
    if model.speed is not None and len(model.speed) >= len(model.true):
        return np.asarray(model.speed[: len(model.true)], dtype=float).reshape(-1)
    return np.linalg.norm(model.true, axis=1)


def omega_norm_values(model: ModelOutputs) -> np.ndarray | None:
    if model.omega is None:
        return None
    return np.linalg.norm(np.asarray(model.omega, dtype=float), axis=1)


def finite_mask(*arrays: np.ndarray | None) -> np.ndarray:
    mask = None
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        m = np.isfinite(arr).all(axis=1) if arr.ndim > 1 else np.isfinite(arr)
        mask = m if mask is None else (mask & m)
    return mask if mask is not None else np.array([], dtype=bool)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 2 and b.ndim == 2:
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
    return float(math.sqrt(mean_squared_error(a, b)))


def error_summary(model: ModelOutputs) -> dict[str, float]:
    err = velocity_errors(model)
    out = {
        "model": model.name,
        "n": len(err),
        "rmse": rmse(model.true, model.pred),
        "mae": float(mean_absolute_error(model.true, model.pred)),
        "median_error": float(np.median(err)),
        "p90_error": float(np.quantile(err, 0.90)),
        "p95_error": float(np.quantile(err, 0.95)),
    }
    for i, comp in enumerate(("vx", "vy", "vz")):
        out[f"rmse_{comp}"] = rmse(model.true[:, i], model.pred[:, i])
        out[f"mae_{comp}"] = float(mean_absolute_error(model.true[:, i], model.pred[:, i]))
    return out


def save_table(df: pd.DataFrame, stem: str, tables_dir: Path) -> None:
    df.to_csv(tables_dir / f"{stem}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    (tables_dir / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def compute_errors(a: ModelOutputs, b: ModelOutputs, tables_dir: Path) -> pd.DataFrame:
    rows = [error_summary(a), error_summary(b)]
    df = pd.DataFrame(rows)
    a_rmse = float(df.loc[df["model"] == a.name, "rmse"].iloc[0])
    b_rmse = float(df.loc[df["model"] == b.name, "rmse"].iloc[0])
    improvement = 100.0 * (b_rmse - a_rmse) / max(b_rmse, 1e-12)
    df["with_flow_relative_improvement_percent"] = np.nan
    df.loc[df["model"] == a.name, "with_flow_relative_improvement_percent"] = improvement
    save_table(df, "basic_performance", tables_dir)
    return df


def slice_definitions(ref: ModelOutputs) -> dict[str, np.ndarray]:
    speed = speed_values(ref)
    lateral = np.linalg.norm(ref.true[:, :2], axis=1)
    vertical = np.abs(ref.true[:, 2])
    values = {
        "speed": speed,
        "lateral_speed": lateral,
        "vertical_speed": vertical,
    }
    if ref.event_density is not None:
        values["event_density"] = np.asarray(ref.event_density, dtype=float).reshape(-1)
    omega_norm = omega_norm_values(ref)
    if omega_norm is not None:
        values["omega_norm"] = omega_norm

    slices: dict[str, np.ndarray] = {"all": np.ones(len(ref.z), dtype=bool)}
    for name, vals in values.items():
        vals = np.asarray(vals, dtype=float)
        lo, hi = np.nanquantile(vals, [0.25, 0.75])
        slices[f"low_{name}"] = vals <= lo
        slices[f"high_{name}"] = vals >= hi
    return slices


def compute_slices(models: list[ModelOutputs], tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    ref = models[0]
    slices = slice_definitions(ref)
    rows = []
    for model in models:
        err = velocity_errors(model)
        for slice_name, mask in slices.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "model": model.name,
                    "slice": slice_name,
                    "n": int(mask.sum()),
                    "rmse": rmse(model.true[mask], model.pred[mask]),
                    "mae": float(mean_absolute_error(model.true[mask], model.pred[mask])),
                    "median_error": float(np.median(err[mask])),
                }
            )
    df = pd.DataFrame(rows)
    save_table(df, "robustness_slices", tables_dir)
    plot_grouped_metric(df, "robustness_slices_rmse", "slice", "rmse", "Velocity RMSE", figures_dir)
    plot_grouped_metric(df, "robustness_slices_mae", "slice", "mae", "Velocity MAE", figures_dir)
    return df


def plot_grouped_metric(
    df: pd.DataFrame, stem: str, x_col: str, y_col: str, ylabel: str, figures_dir: Path
) -> None:
    labels = list(df[x_col].drop_duplicates())
    models = list(df["model"].drop_duplicates())
    x = np.arange(len(labels))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 4.2))
    for j, model in enumerate(models):
        vals = []
        for label in labels:
            row = df[(df["model"] == model) & (df[x_col] == label)]
            vals.append(float(row[y_col].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by slice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png")
    plt.close(fig)


def compute_mahalanobis(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    mask = finite_mask(z)
    if mask.sum() < max(5, z.shape[1] + 1):
        raise ValueError("Not enough finite latent samples to estimate Mahalanobis distance.")
    z_fit = z[mask]
    try:
        cov = LedoitWolf().fit(z_fit)
        mu = cov.location_
        precision = cov.precision_
    except Exception:
        mu = np.nanmean(z_fit, axis=0)
        centered = z_fit - mu
        cov_m = np.cov(centered, rowvar=False)
        shrink = 1e-3 * np.trace(cov_m) / max(cov_m.shape[0], 1)
        precision = np.linalg.pinv(cov_m + shrink * np.eye(cov_m.shape[0]))
    centered_all = z - mu
    d2 = np.einsum("ij,jk,ik->i", centered_all, precision, centered_all)
    return np.sqrt(np.maximum(d2, 0.0))


def reject_option_analysis(
    models: list[ModelOutputs], tables_dir: Path, figures_dir: Path, arrays_dir: Path
) -> pd.DataFrame:
    reject_percentages = np.asarray([0, 1, 2, 5, 10, 15, 20, 30], dtype=float)
    rows = []
    score_cache = {}
    for model in models:
        scores = {
            "latent_norm": np.linalg.norm(model.z, axis=1),
            "mahalanobis": compute_mahalanobis(model.z),
        }
        score_cache[model.name] = scores
        np.savez(arrays_dir / f"{model.name}_reliability_scores.npz", **scores)
        for score_name, score in scores.items():
            order = np.argsort(-score)
            for pct in reject_percentages:
                n_reject = int(round(len(score) * pct / 100.0))
                keep = np.ones(len(score), dtype=bool)
                if n_reject > 0:
                    keep[order[:n_reject]] = False
                rows.append(
                    {
                        "model": model.name,
                        "score": score_name,
                        "reject_percent": pct,
                        "retained_n": int(keep.sum()),
                        "retained_rmse": rmse(model.true[keep], model.pred[keep]),
                        "retained_mae": float(mean_absolute_error(model.true[keep], model.pred[keep])),
                    }
                )
    df = pd.DataFrame(rows)
    save_table(df, "reject_option", tables_dir)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for (model, score), group in df.groupby(["model", "score"]):
        ax.plot(group["reject_percent"], group["retained_rmse"], marker="o", label=f"{model}: {score}")
    ax.set_xlabel("Rejected most-uncertain samples (%)")
    ax.set_ylabel("Retained velocity RMSE")
    ax.set_title("Reject-option reliability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "reject_option_retained_rmse.png")
    plt.close(fig)
    return df


def bootstrap_metric(
    y: np.ndarray,
    score: np.ndarray,
    metric_fn,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    vals = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(float(metric_fn(y[idx], score[idx])))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def bootstrap_spearman(
    err: np.ndarray, score: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    vals = []
    n = len(err)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho = stats.spearmanr(score[idx], err[idx], nan_policy="omit").correlation
        if np.isfinite(rho):
            vals.append(float(rho))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def high_error_detection(
    models: list[ModelOutputs],
    high_error_quantile: float,
    n_bootstrap: int,
    random_seed: int,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_seed)
    for model in models:
        err = velocity_errors(model)
        high = err >= np.quantile(err, high_error_quantile)
        scores = {
            "latent_norm": np.linalg.norm(model.z, axis=1),
            "mahalanobis": compute_mahalanobis(model.z),
        }
        if model.event_density is not None:
            scores["event_density"] = np.asarray(model.event_density, dtype=float).reshape(-1)
        omega_norm = omega_norm_values(model)
        if omega_norm is not None:
            scores["omega_norm"] = omega_norm
        att = attention_group_scores(model)
        for key, val in att.items():
            scores[f"attention_{key}"] = val

        roc_fig, roc_ax = plt.subplots(figsize=(5.2, 4.2))
        pr_fig, pr_ax = plt.subplots(figsize=(5.2, 4.2))
        for score_name, raw_score in scores.items():
            raw_score = np.asarray(raw_score, dtype=float).reshape(-1)
            mask = finite_mask(raw_score, err)
            y = high[mask].astype(int)
            score = raw_score[mask]
            if len(np.unique(y)) < 2 or len(y) < 10:
                continue

            auc_pos = roc_auc_score(y, score)
            auc_neg = roc_auc_score(y, -score)
            if auc_neg > auc_pos:
                score = -score
                sign = -1
                auc = auc_neg
            else:
                sign = 1
                auc = auc_pos
            ap = average_precision_score(y, score)
            rho = stats.spearmanr(score, err[mask], nan_policy="omit").correlation
            auc_lo, auc_hi = bootstrap_metric(y, score, roc_auc_score, n_bootstrap, rng)
            ap_lo, ap_hi = bootstrap_metric(y, score, average_precision_score, n_bootstrap, rng)
            rho_lo, rho_hi = bootstrap_spearman(err[mask], score, n_bootstrap, rng)
            rows.append(
                {
                    "model": model.name,
                    "score": score_name,
                    "chosen_sign": sign,
                    "roc_auc": auc,
                    "roc_auc_ci_low": auc_lo,
                    "roc_auc_ci_high": auc_hi,
                    "pr_auc": ap,
                    "pr_auc_ci_low": ap_lo,
                    "pr_auc_ci_high": ap_hi,
                    "spearman_error": rho,
                    "spearman_ci_low": rho_lo,
                    "spearman_ci_high": rho_hi,
                    "n": int(len(y)),
                    "high_error_n": int(y.sum()),
                }
            )
            fpr, tpr, _ = roc_curve(y, score)
            prec, rec, _ = precision_recall_curve(y, score)
            roc_ax.plot(fpr, tpr, label=f"{score_name} ({auc:.2f})")
            pr_ax.plot(rec, prec, label=f"{score_name} ({ap:.2f})")

        roc_ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("True positive rate")
        roc_ax.set_title(f"High-error ROC: {model.name}")
        roc_ax.legend(fontsize=7)
        roc_fig.tight_layout()
        roc_fig.savefig(figures_dir / f"{model.name}_high_error_roc.png")
        plt.close(roc_fig)

        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.set_title(f"High-error precision-recall: {model.name}")
        pr_ax.legend(fontsize=7)
        pr_fig.tight_layout()
        pr_fig.savefig(figures_dir / f"{model.name}_high_error_pr.png")
        plt.close(pr_fig)

    df = pd.DataFrame(rows)
    save_table(df, "high_error_detection", tables_dir)
    return df


def reliability_score_candidates(model: ModelOutputs) -> dict[str, np.ndarray]:
    """Scores a navigation pipeline could use as risk/fallback indicators."""
    scores = {
        "latent_norm": np.linalg.norm(model.z, axis=1),
        "mahalanobis": compute_mahalanobis(model.z),
    }
    if model.event_density is not None:
        scores["event_density"] = np.asarray(model.event_density, dtype=float).reshape(-1)
    omega_norm = omega_norm_values(model)
    if omega_norm is not None:
        scores["omega_norm"] = omega_norm
    for key, val in attention_group_scores(model).items():
        scores[f"attention_{key}"] = val
    return scores


def orient_risk_score(error: np.ndarray, high: np.ndarray, raw_score: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Orient score so larger values mean higher operational risk."""
    mask = finite_mask(error, raw_score)
    y = high[mask].astype(int)
    score = np.asarray(raw_score, dtype=float).reshape(-1)[mask]
    if len(np.unique(y)) < 2 or len(y) < 10:
        rho = stats.spearmanr(score, error[mask], nan_policy="omit").correlation
        if np.isfinite(rho) and rho < 0:
            return -raw_score, -1, np.nan
        return raw_score, 1, np.nan
    auc_pos = roc_auc_score(y, score)
    auc_neg = roc_auc_score(y, -score)
    if auc_neg > auc_pos:
        return -raw_score, -1, float(auc_neg)
    return raw_score, 1, float(auc_pos)


def navigation_gate_analysis(
    models: list[ModelOutputs],
    high_error_quantile: float,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Evaluate latent/context scores as operational reject-or-fallback gates."""
    reject_percentages = np.asarray([1, 2, 5, 10, 15, 20, 30], dtype=float)
    rows = []
    for model in models:
        err = velocity_errors(model)
        high = err >= np.quantile(err, high_error_quantile)
        high_total = max(int(high.sum()), 1)
        low_total = max(int((~high).sum()), 1)
        for score_name, raw_score in reliability_score_candidates(model).items():
            raw_score = np.asarray(raw_score, dtype=float).reshape(-1)
            risk_score, sign, auc = orient_risk_score(err, high, raw_score)
            order = np.argsort(-risk_score)
            base_rmse = rmse(model.true, model.pred)
            for pct in reject_percentages:
                n_reject = max(1, int(round(len(risk_score) * pct / 100.0)))
                reject_idx = order[:n_reject]
                keep = np.ones(len(risk_score), dtype=bool)
                keep[reject_idx] = False
                rejected_high = int(high[reject_idx].sum())
                rejected_low = int((~high[reject_idx]).sum())
                retained_rmse = rmse(model.true[keep], model.pred[keep])
                rows.append(
                    {
                        "model": model.name,
                        "score": score_name,
                        "risk_sign": sign,
                        "score_roc_auc": auc,
                        "reject_percent": pct,
                        "retained_n": int(keep.sum()),
                        "rejected_n": int(n_reject),
                        "retained_rmse": retained_rmse,
                        "rejected_rmse": rmse(model.true[reject_idx], model.pred[reject_idx]),
                        "rmse_reduction_percent": 100.0 * (base_rmse - retained_rmse) / max(base_rmse, 1e-12),
                        "high_error_recall_rejected": rejected_high / high_total,
                        "rejection_precision_high_error": rejected_high / max(n_reject, 1),
                        "false_alarm_rate_rejected": rejected_low / low_total,
                    }
                )
    df = pd.DataFrame(rows)
    save_table(df, "navigation_gate_policy", tables_dir)

    for metric, ylabel, stem in [
        ("high_error_recall_rejected", "High-error recall in rejected set", "navigation_gate_high_error_recall"),
        ("retained_rmse", "Retained velocity RMSE", "navigation_gate_retained_rmse"),
        ("rejection_precision_high_error", "Precision of rejected set", "navigation_gate_precision"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for (model_name, score_name), group in df.groupby(["model", "score"]):
            if score_name not in {"mahalanobis", "latent_norm", "event_density", "omega_norm"}:
                continue
            group = group.sort_values("reject_percent")
            ax.plot(group["reject_percent"], group[metric], marker="o", label=f"{model_name}: {score_name}")
        ax.set_xlabel("Rejected / fallback samples (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs operational gate threshold")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{stem}.png")
        plt.close(fig)
    return df


def reliability_curves(models: list[ModelOutputs], tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows = []
    for model in models:
        err = velocity_errors(model)
        scores = {
            "latent_norm": np.linalg.norm(model.z, axis=1),
            "mahalanobis": compute_mahalanobis(model.z),
        }
        for score_name, score in scores.items():
            df_tmp = pd.DataFrame({"score": score, "error": err})
            df_tmp = df_tmp.replace([np.inf, -np.inf], np.nan).dropna()
            df_tmp["bin"] = pd.qcut(df_tmp["score"], q=10, labels=False, duplicates="drop")
            for bin_id, group in df_tmp.groupby("bin"):
                rows.append(
                    {
                        "model": model.name,
                        "score": score_name,
                        "bin": int(bin_id),
                        "score_min": float(group["score"].min()),
                        "score_max": float(group["score"].max()),
                        "score_mean": float(group["score"].mean()),
                        "mean_error": float(group["error"].mean()),
                        "median_error": float(group["error"].median()),
                        "p90_error": float(group["error"].quantile(0.90)),
                        "q25_error": float(group["error"].quantile(0.25)),
                        "q75_error": float(group["error"].quantile(0.75)),
                        "n": int(len(group)),
                    }
                )
    df = pd.DataFrame(rows)
    save_table(df, "reliability_bins", tables_dir)

    for score_name in df["score"].drop_duplicates():
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        for model_name, group in df[df["score"] == score_name].groupby("model"):
            group = group.sort_values("bin")
            lower = np.maximum(group["mean_error"].to_numpy() - group["q25_error"].to_numpy(), 0.0)
            upper = np.maximum(group["q75_error"].to_numpy() - group["mean_error"].to_numpy(), 0.0)
            yerr = np.vstack([lower, upper])
            ax.errorbar(group["bin"], group["mean_error"], yerr=yerr, marker="o", capsize=3, label=model_name)
            ax.plot(group["bin"], group["median_error"], linestyle="--", alpha=0.75)
        ax.set_xlabel(f"{score_name} quantile bin")
        ax.set_ylabel("Velocity error")
        ax.set_title(f"Reliability curve: {score_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"reliability_curve_{score_name}.png")
        plt.close(fig)
    return df


def pca_compactness(z: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=float)
    z = z - np.nanmean(z, axis=0, keepdims=True)
    pca = PCA(n_components=min(z.shape[0], z.shape[1], 32))
    pca.fit(z)
    ev = pca.explained_variance_ratio_
    eig = pca.explained_variance_
    csum = np.cumsum(ev)
    return {
        "pca_dim_90": float(np.searchsorted(csum, 0.90) + 1),
        "pca_dim_95": float(np.searchsorted(csum, 0.95) + 1),
        "pc1_explained_variance": float(ev[0]) if len(ev) else np.nan,
        "pc2_explained_variance": float(ev[1]) if len(ev) > 1 else np.nan,
        "participation_ratio": float((eig.sum() ** 2) / np.maximum(np.sum(eig ** 2), 1e-12)),
    }


def metric_lookup(
    df: pd.DataFrame,
    model: str,
    analysis: str | None = None,
    target: str | None = None,
    metric: str | None = None,
) -> float:
    if df.empty:
        return np.nan
    sub = df[df["model"] == model]
    if analysis is not None and "analysis" in sub:
        sub = sub[sub["analysis"] == analysis]
    if target is not None and "target" in sub:
        sub = sub[sub["target"] == target]
    if metric is not None and "metric" in sub:
        sub = sub[sub["metric"] == metric]
    if len(sub) == 0:
        return np.nan
    return float(sub["value"].iloc[0])


def navigation_readiness_summary(
    models: list[ModelOutputs],
    basic_df: pd.DataFrame,
    high_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    """Summarize whether the latent is useful as a navigation-pipeline signal."""
    rows = []
    for model in models:
        basic = basic_df[basic_df["model"] == model.name].iloc[0]
        compact = pca_compactness(model.z)
        best_auc = np.nan
        best_score = ""
        if len(high_df):
            sub = high_df[high_df["model"] == model.name]
            if len(sub):
                row = sub.sort_values("roc_auc", ascending=False).iloc[0]
                best_auc = float(row["roc_auc"])
                best_score = str(row["score"])
        gate10 = pd.DataFrame()
        if len(gate_df):
            gate_sub = gate_df[(gate_df["model"] == model.name) & (gate_df["reject_percent"] == 10)]
            if len(gate_sub):
                gate10 = gate_sub.sort_values("high_error_recall_rejected", ascending=False).head(1)

        def add(category: str, metric: str, value: float, use: str, caveat: str = "") -> None:
            rows.append(
                {
                    "model": model.name,
                    "category": category,
                    "metric": metric,
                    "value": value,
                    "navigation_use": use,
                    "caveat": caveat,
                }
            )

        add("estimation", "velocity_rmse", float(basic["rmse"]), "Nominal estimator accuracy.")
        add("estimation", "p90_velocity_error", float(basic["p90_error"]), "Tail error budget for safety margins.")
        add("compact_state", "pca_dim_95", compact["pca_dim_95"], "Size of a low-dimensional latent monitor/state.")
        add("compact_state", "participation_ratio", compact["participation_ratio"], "Effective latent dimensionality.")
        add(
            "physical_readout",
            "ridge_z_to_velocity_r2",
            metric_lookup(physical_df, model.name, "ridge_probe", "velocity", "r2"),
            "Can a simple downstream navigation module decode velocity from z?",
        )
        add(
            "physical_readout",
            "ridge_z_to_velocity_rmse",
            metric_lookup(physical_df, model.name, "ridge_probe", "velocity", "rmse"),
            "Error if z is used by a lightweight linear readout.",
        )
        add(
            "physical_alignment",
            "cca_top1_z_vs_velocity_speed",
            metric_lookup(physical_df, model.name, "cca", "canonical_1", "correlation"),
            "Checks whether latent axes align with physical motion variables.",
        )
        add(
            "risk_monitoring",
            "best_high_error_roc_auc",
            best_auc,
            f"Best available high-error detector score: {best_score}.",
            "AUC is not calibrated probability.",
        )
        if len(gate10):
            row = gate10.iloc[0]
            add(
                "operational_gate",
                "best_score_reject10_high_error_recall",
                float(row["high_error_recall_rejected"]),
                f"Fraction of high-error windows caught by rejecting/falling back on top 10% `{row['score']}` risk.",
                "Rejected samples need a fallback estimator or conservative mode.",
            )
            add(
                "operational_gate",
                "best_score_reject10_retained_rmse",
                float(row["retained_rmse"]),
                f"Retained RMSE after top 10% `{row['score']}` risk gate.",
            )
        if len(temporal_df):
            tsub = temporal_df[temporal_df["model"] == model.name]
            if len(tsub):
                add(
                    "temporal_state",
                    "mean_latent_smoothness",
                    float(tsub["latent_smoothness"].mean()),
                    "Smoothness of z trajectories for filter-like state propagation.",
                    "Only valid when samples are sequence-contiguous.",
                )
                if "lag_1" in tsub:
                    add(
                        "temporal_state",
                        "mean_residual_autocorr_lag1",
                        float(tsub["lag_1"].mean()),
                        "Residual temporal correlation relevant to innovation gating.",
                    )
    df = pd.DataFrame(rows)
    save_table(df, "navigation_readiness_summary", tables_dir)
    return df


def attention_group_scores(model: ModelOutputs) -> dict[str, np.ndarray]:
    if model.attention is None:
        return {}
    att = np.asarray(model.attention, dtype=float)
    if att.ndim == 4:
        att = att.mean(axis=1)
    if att.ndim != 3 or att.shape[-1] < 7 or att.shape[-2] < 7:
        warnings.warn(f"{model.name}: attention shape {att.shape} is not compatible with 7-token analysis.")
        return {}
    out = {}
    for name, idxs in TOKEN_GROUPS.items():
        out[name] = att[:, :, idxs].mean(axis=(1, 2))
    return out


def binned_line_data(x: np.ndarray, y: np.ndarray, bins: int = 8) -> pd.DataFrame:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return pd.DataFrame()
    df["bin"] = pd.qcut(df["x"], q=bins, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in df.groupby("bin"):
        rows.append(
            {
                "bin": int(bin_id),
                "x_mean": float(group["x"].mean()),
                "y_mean": float(group["y"].mean()),
                "y_q25": float(group["y"].quantile(0.25)),
                "y_q75": float(group["y"].quantile(0.75)),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def attention_analysis(models: list[ModelOutputs], tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows = []
    bin_rows = []
    for model in models:
        groups = attention_group_scores(model)
        if not groups:
            warnings.warn(f"{model.name}: attention arrays missing; skipping attention analysis.")
            continue
        err = velocity_errors(model)
        context = {
            "error": err,
            "speed": speed_values(model),
        }
        if model.event_density is not None:
            context["event_density"] = np.asarray(model.event_density, dtype=float).reshape(-1)
        omega_norm = omega_norm_values(model)
        if omega_norm is not None:
            context["omega_norm"] = omega_norm

        for group_name, weights in groups.items():
            for ctx_name, ctx_vals in context.items():
                mask = finite_mask(weights, ctx_vals)
                if mask.sum() < 5:
                    continue
                rho = stats.spearmanr(weights[mask], ctx_vals[mask], nan_policy="omit").correlation
                rows.append(
                    {
                        "model": model.name,
                        "attention_group": group_name,
                        "context": ctx_name,
                        "spearman": float(rho) if np.isfinite(rho) else np.nan,
                        "n": int(mask.sum()),
                    }
                )
                bd = binned_line_data(ctx_vals[mask], weights[mask], bins=8)
                for _, row in bd.iterrows():
                    bin_rows.append(
                        {
                            "model": model.name,
                            "attention_group": group_name,
                            "context": ctx_name,
                            **row.to_dict(),
                        }
                    )

    corr_df = pd.DataFrame(rows)
    bins_df = pd.DataFrame(bin_rows)
    if len(corr_df):
        save_table(corr_df, "attention_correlations", tables_dir)
    if len(bins_df):
        save_table(bins_df, "attention_binned", tables_dir)
        for ctx_name in bins_df["context"].drop_duplicates():
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            for (model_name, group_name), group in bins_df[bins_df["context"] == ctx_name].groupby(
                ["model", "attention_group"]
            ):
                group = group.sort_values("bin")
                ax.plot(group["x_mean"], group["y_mean"], marker="o", label=f"{model_name}: {group_name}")
            ax.set_xlabel(ctx_name.replace("_", " "))
            ax.set_ylabel("Mean source attention")
            ax.set_title(f"Attention groups vs {ctx_name.replace('_', ' ')}")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(figures_dir / f"attention_vs_{ctx_name}.png")
            plt.close(fig)
    return corr_df


def ridge_probe(
    x: np.ndarray, y: np.ndarray, random_seed: int, test_size: float = 0.25
) -> dict[str, float]:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_seed
    )
    sx = StandardScaler().fit(x_train)
    sy = StandardScaler().fit(y_train)
    xtr = sx.transform(x_train)
    xte = sx.transform(x_test)
    ytr = sy.transform(y_train)
    model = Ridge(alpha=1.0)
    model.fit(xtr, ytr)
    pred_scaled = model.predict(xte)
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(-1, 1)
    pred = sy.inverse_transform(pred_scaled)
    return {
        "r2": float(r2_score(y_test, pred, multioutput="uniform_average")),
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def physical_alignment(
    models: list[ModelOutputs], tables_dir: Path, figures_dir: Path, random_seed: int
) -> pd.DataFrame:
    rows = []
    for model in models:
        y_phys = np.column_stack([model.true, speed_values(model)])
        mask = finite_mask(model.z, y_phys)
        z = model.z[mask]
        y = y_phys[mask]
        n_components = min(4, z.shape[1], y.shape[1])
        sx = StandardScaler().fit_transform(z)
        sy = StandardScaler().fit_transform(y)
        cca = CCA(n_components=n_components, max_iter=2000)
        zx, zy = cca.fit_transform(sx, sy)
        corrs = []
        for i in range(n_components):
            corrs.append(float(np.corrcoef(zx[:, i], zy[:, i])[0, 1]))
            rows.append(
                {
                    "model": model.name,
                    "analysis": "cca",
                    "target": f"canonical_{i+1}",
                    "metric": "correlation",
                    "value": corrs[-1],
                    "n": int(len(z)),
                }
            )
        for target_name, target in [
            ("velocity", model.true),
            ("speed", speed_values(model).reshape(-1, 1)),
        ]:
            probe = ridge_probe(model.z, target, random_seed)
            for metric, value in probe.items():
                rows.append(
                    {
                        "model": model.name,
                        "analysis": "ridge_probe",
                        "target": target_name,
                        "metric": metric,
                        "value": value,
                        "n": int(len(model.z)),
                    }
                )

        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.bar(np.arange(1, len(corrs) + 1), corrs)
        ax.set_xlabel("Canonical component")
        ax.set_ylabel("Canonical correlation")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"CCA: latent vs physical variables ({model.name})")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{model.name}_cca_correlations.png")
        plt.close(fig)

    df = pd.DataFrame(rows)
    save_table(df, "physical_alignment", tables_dir)
    return df


def temporal_available(model: ModelOutputs) -> bool:
    return model.seq is not None and model.times is not None


def temporal_pairs(model: ModelOutputs) -> tuple[np.ndarray, np.ndarray]:
    seq = model.seq.reshape(-1)
    times = model.times.reshape(-1).astype(float)
    left, right = [], []
    for sid in pd.unique(seq):
        idx = np.where(seq == sid)[0]
        idx = idx[np.argsort(times[idx])]
        if len(idx) < 2:
            continue
        left.extend(idx[:-1])
        right.extend(idx[1:])
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def residual_autocorr(residual_norm: np.ndarray, max_lag: int = 10) -> dict[str, float]:
    out = {}
    for lag in range(1, max_lag + 1):
        if len(residual_norm) <= lag + 2:
            out[f"lag_{lag}"] = np.nan
        else:
            out[f"lag_{lag}"] = float(np.corrcoef(residual_norm[:-lag], residual_norm[lag:])[0, 1])
    return out


def temporal_diagnostics(
    models: list[ModelOutputs],
    tables_dir: Path,
    figures_dir: Path,
    random_seed: int,
    require_temporal: bool,
) -> pd.DataFrame:
    if require_temporal and not all(temporal_available(m) for m in models):
        missing = [m.name for m in models if not temporal_available(m)]
        raise ValueError(f"--temporal-only was set, but sequence IDs/timestamps are missing for {missing}.")

    rows = []
    seq_rows = []
    for model in models:
        if not temporal_available(model):
            warnings.warn(f"{model.name}: sequence IDs/timestamps missing; skipping temporal diagnostics.")
            continue
        seq = model.seq.reshape(-1)
        times = model.times.reshape(-1).astype(float)
        errors = velocity_errors(model)
        for sid in pd.unique(seq):
            idx = np.where(seq == sid)[0]
            idx = idx[np.argsort(times[idx])]
            if len(idx) < 3:
                continue
            z = model.z[idx]
            v = model.true[idx]
            pred = model.pred[idx]
            dz2 = z[2:] - 2 * z[1:-1] + z[:-2]
            step = z[1:] - z[:-1]
            dv = v[1:] - v[:-1]
            smooth = float(np.linalg.norm(dz2, axis=1).mean())
            step_norm = np.linalg.norm(step, axis=1)
            dv_norm = np.linalg.norm(dv, axis=1)
            rho = stats.spearmanr(step_norm, dv_norm, nan_policy="omit").correlation
            resid = np.linalg.norm(pred - v, axis=1)
            ac = residual_autocorr(resid, max_lag=10)
            seq_rows.append(
                {
                    "model": model.name,
                    "sequence": str(sid),
                    "n": int(len(idx)),
                    "rmse": rmse(v, pred),
                    "mae": float(mean_absolute_error(v, pred)),
                    "latent_smoothness": smooth,
                    "mean_step_norm": float(step_norm.mean()),
                    "spearman_step_vs_dv": float(rho) if np.isfinite(rho) else np.nan,
                    **ac,
                }
            )
        left, right = temporal_pairs(model)
        if len(left) > 10:
            for target_name, target in [("z_next", model.z[right]), ("v_next", model.true[right])]:
                probe = ridge_probe(model.z[left], target, random_seed)
                for metric, value in probe.items():
                    rows.append(
                        {
                            "model": model.name,
                            "analysis": "temporal_transition_probe",
                            "target": target_name,
                            "metric": metric,
                            "value": value,
                            "n": int(len(left)),
                        }
                    )

    seq_df = pd.DataFrame(seq_rows)
    probe_df = pd.DataFrame(rows)
    if len(seq_df):
        save_table(seq_df, "temporal_sequence_metrics", tables_dir)
        plot_temporal_examples(models, seq_df, figures_dir)
    if len(probe_df):
        save_table(probe_df, "temporal_transition_probes", tables_dir)
    return seq_df


def plot_temporal_examples(models: list[ModelOutputs], seq_df: pd.DataFrame, figures_dir: Path) -> None:
    if len(models) < 2 or seq_df.empty:
        return
    pivot = seq_df.pivot(index="sequence", columns="model", values="rmse").dropna()
    if len(pivot) == 0:
        return
    diff = pivot[models[0].name] - pivot[models[1].name]
    selected = [diff.idxmin(), diff.idxmax()]
    for seq_id in dict.fromkeys(selected):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for model in models:
            seq = model.seq.reshape(-1)
            times = model.times.reshape(-1).astype(float)
            idx = np.where(seq.astype(str) == str(seq_id))[0]
            if len(idx) == 0:
                continue
            idx = idx[np.argsort(times[idx])]
            ax.plot(times[idx], velocity_errors(model)[idx], label=model.name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity error")
        ax.set_title(f"Sequence {seq_id}: error over time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"temporal_sequence_{seq_id}_error.png")
        plt.close(fig)


def color_contexts(model: ModelOutputs) -> dict[str, np.ndarray]:
    ctx = {
        "speed": speed_values(model),
        "error": velocity_errors(model),
    }
    if model.event_density is not None:
        ctx["event_density"] = np.asarray(model.event_density, dtype=float).reshape(-1)
    omega_norm = omega_norm_values(model)
    if omega_norm is not None:
        ctx["omega_norm"] = omega_norm
    return ctx


def scatter_pca_2d(z2: np.ndarray, color: np.ndarray, title: str, cbar: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=color, s=7, alpha=0.75, cmap="viridis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(cbar)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def scatter_pca_3d(z3: np.ndarray, color: np.ndarray, title: str, cbar: str, out: Path) -> None:
    fig = plt.figure(figsize=(5.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(z3[:, 0], z3[:, 1], z3[:, 2], c=color, s=6, alpha=0.7, cmap="viridis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75)
    cb.set_label(cbar)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def manifold_plots(
    models: list[ModelOutputs], figures_dir: Path, arrays_dir: Path, make_umap: bool, random_seed: int
) -> None:
    for model in models:
        z = StandardScaler().fit_transform(model.z)
        pca = PCA(n_components=min(3, z.shape[1]), random_state=random_seed)
        zp = pca.fit_transform(z)
        np.savez(arrays_dir / f"{model.name}_pca_embedding.npz", embedding=zp, explained_variance=pca.explained_variance_ratio_)
        contexts = color_contexts(model)
        for cname, cval in contexts.items():
            scatter_pca_2d(
                zp[:, :2],
                cval,
                f"{model.name}: PCA colored by {cname}",
                cname,
                figures_dir / f"{model.name}_pca2d_{cname}.png",
            )
            if zp.shape[1] >= 3:
                scatter_pca_3d(
                    zp[:, :3],
                    cval,
                    f"{model.name}: 3D PCA colored by {cname}",
                    cname,
                    figures_dir / f"{model.name}_pca3d_{cname}.png",
                )

        if make_umap:
            try:
                import umap
            except Exception as exc:
                warnings.warn(f"UMAP requested but umap-learn is unavailable: {exc}")
                continue
            reducer = umap.UMAP(n_components=2, random_state=random_seed)
            zu = reducer.fit_transform(z)
            np.savez(arrays_dir / f"{model.name}_umap_embedding.npz", embedding=zu)
            for cname, cval in contexts.items():
                scatter_pca_2d(
                    zu,
                    cval,
                    f"{model.name}: UMAP colored by {cname}",
                    cname,
                    figures_dir / f"{model.name}_umap2d_{cname}.png",
                )

    # Joint PCA for direct visual comparison.
    zcat = np.concatenate([m.z for m in models], axis=0)
    labels = np.concatenate([[m.name] * len(m.z) for m in models])
    zcat = StandardScaler().fit_transform(zcat)
    joint = PCA(n_components=2, random_state=random_seed).fit_transform(zcat)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for model in models:
        mask = labels == model.name
        ax.scatter(joint[mask, 0], joint[mask, 1], s=7, alpha=0.55, label=model.name)
    ax.set_xlabel("Joint PC1")
    ax.set_ylabel("Joint PC2")
    ax.set_title("Joint PCA comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "joint_pca_model_labels.png")
    plt.close(fig)


def label_model(name: str) -> str:
    return name.replace("_best", "").replace("_", " ")


def plot_paper_robustness(slice_df: pd.DataFrame, models: list[ModelOutputs], out_dir: Path) -> None:
    desired = [
        ("all", "All samples"),
        ("low_event_density", "Sparse events"),
        ("high_speed", "High speed"),
        ("high_lateral_speed", "High lateral speed"),
        ("high_vertical_speed", "High vertical speed"),
        ("high_omega_norm", "High angular rate"),
    ]
    available = {s for s in slice_df["slice"].unique()}
    selected = [(k, lab) for k, lab in desired if k in available]
    if not selected:
        return
    y = np.arange(len(selected))
    width = 0.34
    colors = ["#2563eb", "#f97316"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for j, model in enumerate(models):
        vals = []
        for key, _ in selected:
            row = slice_df[(slice_df["model"] == model.name) & (slice_df["slice"] == key)]
            vals.append(float(row["rmse"].iloc[0]) if len(row) else np.nan)
        ax.barh(y + (j - 0.5) * width, vals, width, label=label_model(model.name), color=colors[j % len(colors)])
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in selected])
    ax.invert_yaxis()
    ax.set_xlabel("Velocity RMSE")
    ax.set_title("Velocity error in navigation-relevant regimes")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_figure_1_robustness_summary.png")
    plt.close(fig)


def best_gate_score(gate_df: pd.DataFrame, model_name: str, reject_percent: float = 10.0) -> str | None:
    sub = gate_df[(gate_df["model"] == model_name) & (gate_df["reject_percent"] == reject_percent)]
    if len(sub) == 0:
        return None
    return str(sub.sort_values("high_error_recall_rejected", ascending=False).iloc[0]["score"])


def plot_paper_navigation_gate(gate_df: pd.DataFrame, models: list[ModelOutputs], out_dir: Path) -> None:
    if gate_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharex=True)
    colors = ["#2563eb", "#f97316"]
    for j, model in enumerate(models):
        score = best_gate_score(gate_df, model.name) or "mahalanobis"
        sub = gate_df[(gate_df["model"] == model.name) & (gate_df["score"] == score)].sort_values("reject_percent")
        if len(sub) == 0:
            continue
        label = f"{label_model(model.name)} ({score})"
        axes[0].plot(sub["reject_percent"], sub["retained_rmse"], marker="o", color=colors[j % len(colors)], label=label)
        axes[1].plot(
            sub["reject_percent"],
            sub["high_error_recall_rejected"],
            marker="o",
            color=colors[j % len(colors)],
            label=label,
        )
    axes[0].set_ylabel("Retained velocity RMSE")
    axes[1].set_ylabel("High-error recall")
    for ax in axes:
        ax.set_xlabel("Samples routed to fallback (%)")
        ax.grid(alpha=0.25)
    axes[0].set_title("Accuracy after risk gating")
    axes[1].set_title("High-error windows caught")
    axes[1].set_ylim(0, 1)
    axes[0].legend()
    if SHOW_FIGURE_TITLES:
        fig.suptitle("Latent-derived gate for fallback navigation", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_figure_2_navigation_gate.png")
    plt.close(fig)


def trajectory_progress(model: ModelOutputs) -> np.ndarray | None:
    """Return normalized within-sequence sample order in [0, 1], if possible."""
    if model.seq is None:
        return None
    seq = model.seq.reshape(-1)
    if model.times is not None:
        times = model.times.reshape(-1).astype(float)
    else:
        times = np.arange(len(seq), dtype=float)
    progress = np.full(len(seq), np.nan, dtype=float)
    for sid in pd.unique(seq):
        idx = np.where(seq == sid)[0]
        # Stable sort preserves file order for repeated timestamps.
        order = idx[np.argsort(times[idx], kind="mergesort")]
        denom = max(len(order) - 1, 1)
        progress[order] = np.arange(len(order), dtype=float) / denom
    return progress


def plot_paper_risk_calibration(reliability_df: pd.DataFrame, models: list[ModelOutputs], out_dir: Path) -> None:
    if reliability_df.empty:
        return
    score = "mahalanobis"
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    colors = ["#2563eb", "#f97316"]
    for j, model in enumerate(models):
        sub = reliability_df[(reliability_df["model"] == model.name) & (reliability_df["score"] == score)].sort_values("bin")
        if len(sub) == 0:
            continue
        x = sub["bin"].to_numpy()
        mean = sub["mean_error"].to_numpy()
        q25 = sub["q25_error"].to_numpy()
        q75 = sub["q75_error"].to_numpy()
        color = colors[j % len(colors)]
        ax.plot(x, mean, marker="o", linewidth=2, color=color, label=label_model(model.name))
        ax.fill_between(x, q25, q75, color=color, alpha=0.15)
    ax.set_xlabel("Mahalanobis risk decile")
    ax.set_ylabel("Velocity error")
    if SHOW_FIGURE_TITLES:
        ax.set_title("Latent distance behaves as a risk indicator")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_figure_3_risk_calibration.png")
    plt.close(fig)


def plot_paper_latent_manifold(model: ModelOutputs, out_dir: Path) -> None:
    z = np.asarray(model.z, dtype=float)
    z = z - np.nanmean(z, axis=0, keepdims=True)
    emb = PCA(n_components=2, random_state=42).fit_transform(z)
    speed = speed_values(model)
    err = velocity_errors(model)
    high = err >= np.quantile(err, 0.90)
    progress = trajectory_progress(model)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharex=True, sharey=True)
    panels = [("Speed", speed, "viridis")]
    if progress is not None and np.isfinite(progress).any():
        panels.append(("Trajectory progress", progress, "plasma"))
    else:
        panels.append(("Prediction error", err, "magma"))

    for ax, (label, values, cmap) in zip(axes, panels):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=8, cmap=cmap, alpha=0.72, linewidths=0)
        if high.any() and label == "Speed":
            ax.scatter(
                emb[high, 0],
                emb[high, 1],
                facecolors="none",
                edgecolors="#ef4444",
                s=26,
                linewidths=0.75,
                label="Top 10% error",
            )
            # Keep the high-error key outside the data region: the manifold
            # branches occupy nearly the full first panel.
            handles, labels = ax.get_legend_handles_labels()
        ax.set_xlabel("PC1")
        ax.set_title(label)
        ax.grid(alpha=0.20)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(label)
    axes[0].set_ylabel("PC2")
    if SHOW_FIGURE_TITLES:
        fig.suptitle(f"Latent manifold: speed structure versus trajectory phase ({label_model(model.name)})", y=.99)
    if high.any() and handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=1)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_dir / "paper_figure_4_latent_manifold.png")
    plt.close(fig)

    stats_rows = []
    fields = {"speed": speed, "error": err}
    if progress is not None:
        fields["trajectory_progress"] = progress
    for name, values in fields.items():
        mask = finite_mask(values, emb)
        if mask.sum() < 5:
            continue
        stats_rows.append(
            {
                "variable": name,
                "spearman_pc1": float(stats.spearmanr(emb[mask, 0], values[mask], nan_policy="omit").correlation),
                "spearman_pc2": float(stats.spearmanr(emb[mask, 1], values[mask], nan_policy="omit").correlation),
                "pearson_pc1": float(np.corrcoef(emb[mask, 0], values[mask])[0, 1]),
                "pearson_pc2": float(np.corrcoef(emb[mask, 1], values[mask])[0, 1]),
            }
        )
    if progress is not None:
        mask = finite_mask(speed, progress)
        stats_rows.append(
            {
                "variable": "speed_vs_trajectory_progress",
                "spearman_pc1": float(stats.spearmanr(speed[mask], progress[mask], nan_policy="omit").correlation),
                "spearman_pc2": np.nan,
                "pearson_pc1": float(np.corrcoef(speed[mask], progress[mask])[0, 1]),
                "pearson_pc2": np.nan,
            }
        )
    pd.DataFrame(stats_rows).to_csv(out_dir / "paper_figure_4_phase_correlations.csv", index=False)


def write_paper_plot_interpretation(
    out_dir: Path,
    basic_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    navigation_df: pd.DataFrame,
    models: list[ModelOutputs],
) -> None:
    with_name = models[0].name
    without_name = models[1].name
    wf = basic_df[basic_df["model"] == with_name].iloc[0]
    nf = basic_df[basic_df["model"] == without_name].iloc[0]
    gain = 100.0 * (nf["rmse"] - wf["rmse"]) / max(float(nf["rmse"]), 1e-12)

    def nav_value(model: str, metric: str) -> float:
        row = navigation_df[(navigation_df["model"] == model) & (navigation_df["metric"] == metric)]
        return float(row["value"].iloc[0]) if len(row) else np.nan

    gate_score = best_gate_score(gate_df, with_name) or "mahalanobis"
    gate10 = gate_df[
        (gate_df["model"] == with_name) & (gate_df["score"] == gate_score) & (gate_df["reject_percent"] == 10)
    ]
    gate_text = ""
    if len(gate10):
        g = gate10.iloc[0]
        gate_text = (
            f"At a 10% fallback rate, `{gate_score}` reduces retained RMSE to "
            f"{g['retained_rmse']:.2f} and catches {100*g['high_error_recall_rejected']:.1f}% "
            "of high-error windows."
        )

    progress = trajectory_progress(models[0])
    speed = speed_values(models[0])
    z = np.asarray(models[0].z, dtype=float)
    z = z - np.nanmean(z, axis=0, keepdims=True)
    emb = PCA(n_components=2, random_state=42).fit_transform(z)
    phase_text = ""
    if progress is not None:
        mask = finite_mask(speed, progress, emb)
        rho_speed_progress = stats.spearmanr(speed[mask], progress[mask], nan_policy="omit").correlation
        rho_pc2_speed = stats.spearmanr(emb[mask, 1], speed[mask], nan_policy="omit").correlation
        rho_pc2_progress = stats.spearmanr(emb[mask, 1], progress[mask], nan_policy="omit").correlation
        phase_text = (
            f"In this run, speed is strongly organized along PC2 (Spearman {rho_pc2_speed:.2f}), "
            f"whereas trajectory progress is weakly related to PC2 (Spearman {rho_pc2_progress:.2f}) "
            f"and speed is only weakly related to progress overall (Spearman {rho_speed_progress:.2f}). "
            "Thus the visible speed gradient should not be described as only an encoding of sample order."
        )

    lines = [
        "# Curated Paper Figure Interpretation",
        "",
        "These four figures are intended as the paper-facing subset. The remaining figures and tables are diagnostic support material.",
        "",
        "## Figure 1 - Robustness Summary",
        "",
        "**Critical evaluation.** This is the cleanest plot for demonstrating practical estimation gain. It compares velocity RMSE in regimes that matter to navigation: sparse events, high speed, high lateral speed, vertical motion, and high angular excitation when available.",
        "",
        f"**Interpretation.** The with-flow model reduces global RMSE from {nf['rmse']:.2f} to {wf['rmse']:.2f}, a {gain:.1f}% relative improvement. The important point is not only the average gain, but whether the gain persists in hard regimes. If the sparse-event/high-speed bars remain lower for with-flow, the auxiliary task is improving motion evidence where event-based navigation is most fragile.",
        "",
        "**Caveat.** Slices are quantile-defined on this evaluation set. They are operationally useful stress tests, but not universal flight-envelope thresholds.",
        "",
        "## Figure 2 - Latent-Derived Navigation Gate",
        "",
        "**Critical evaluation.** This figure answers whether the latent can drive a fallback policy. The left panel shows the accuracy retained after routing the riskiest windows away from the learned estimator. The right panel shows whether those rejected windows actually contain high-error cases.",
        "",
        f"**Interpretation.** {gate_text} A useful navigation monitor should reduce retained RMSE without rejecting arbitrary samples. Mahalanobis distance is especially interpretable here because it measures how far a window lies from the learned latent distribution.",
        "",
        "**Caveat.** This is not calibrated uncertainty. It is a ranking signal for fallback, covariance inflation, or conservative control.",
        "",
        "## Figure 3 - Risk Calibration By Latent Distance",
        "",
        "**Critical evaluation.** This plot checks monotonicity: as latent Mahalanobis distance increases, prediction error should rise. That is the key property needed for a practical risk monitor.",
        "",
        "**Interpretation.** A rising curve means latent geometry contains reliability information, not just task information. The interquartile band is important: wide bands indicate that the score is useful statistically but not deterministic at the single-sample level.",
        "",
        "**Caveat.** If high deciles flatten or become noisy, the latent score should be used only as one cue together with event density, IMU excitation, and classical residual checks.",
        "",
        "## Figure 4 - Compact Latent Manifold With Failure Overlay",
        "",
        "**Critical evaluation.** This is the best visual explanation of the latent as a navigation state. It now compares the same PCA manifold colored by physical speed and by normalized within-trajectory progress, so it directly checks whether the speed structure is merely a trajectory-phase artifact.",
        "",
        f"**Interpretation.** The with-flow latent is compact: `pca_dim_95={nav_value(with_name, 'pca_dim_95'):.0f}` and participation ratio `{nav_value(with_name, 'participation_ratio'):.2f}` in the summary table. A coherent speed gradient supports using the latent as a compact state-like representation. {phase_text} Red high-error rings reveal where the learned state is less reliable.",
        "",
        "**Caveat.** PCA is a 2D projection and speed is itself physically correlated with landing phase in some trajectories. Use the side-by-side progress panel and `paper_figure_4_phase_correlations.csv` to avoid over-claiming that the network has learned speed independently of trajectory phase.",
        "",
    ]
    (out_dir / "paper_plot_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def curated_paper_plots(
    models: list[ModelOutputs],
    basic_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    navigation_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paper_robustness(slice_df, models, out_dir)
    plot_paper_navigation_gate(gate_df, models, out_dir)
    plot_paper_risk_calibration(reliability_df, models, out_dir)
    plot_paper_latent_manifold(models[0], out_dir)
    write_paper_plot_interpretation(out_dir, basic_df, slice_df, gate_df, reliability_df, navigation_df, models)


def strongest_gain_table(slice_df: pd.DataFrame, with_name: str, without_name: str) -> pd.DataFrame:
    pivot = slice_df.pivot(index="slice", columns="model", values="rmse").dropna()
    if with_name not in pivot or without_name not in pivot:
        return pd.DataFrame()
    out = pivot.reset_index()
    out["rmse_delta_with_minus_without"] = out[with_name] - out[without_name]
    out["relative_gain_percent"] = 100.0 * (out[without_name] - out[with_name]) / np.maximum(out[without_name], 1e-12)
    return out.sort_values("rmse_delta_with_minus_without")


def write_report(
    out_dir: Path,
    basic_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    reject_df: pd.DataFrame,
    high_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    navigation_df: pd.DataFrame,
    align_mode: str,
    models: list[ModelOutputs],
) -> None:
    with_name = models[0].name
    without_name = models[1].name
    lines = ["# Journal Latent Analysis Report", ""]
    lines.append(f"- Alignment mode: `{align_mode}`")
    lines.append(f"- Aligned samples: `{len(models[0].z)}`")
    lines.append("- Curated paper figures: `figures_paper/`")
    lines.append("- Curated interpretation notes: `figures_paper/paper_plot_interpretation.md`")
    lines.append("")

    wf = basic_df[basic_df["model"] == with_name].iloc[0]
    nf = basic_df[basic_df["model"] == without_name].iloc[0]
    gain = 100.0 * (nf["rmse"] - wf["rmse"]) / max(float(nf["rmse"]), 1e-12)
    lines += [
        "## Key Performance Change",
        "",
        f"- `{with_name}` RMSE: `{wf['rmse']:.6f}`",
        f"- `{without_name}` RMSE: `{nf['rmse']:.6f}`",
        f"- Relative RMSE improvement: `{gain:.2f}%`",
        "",
    ]

    lines += ["## Navigation-Pipeline Use Of The Latent Space", ""]
    if len(navigation_df):
        for model_name in navigation_df["model"].drop_duplicates():
            lines.append(f"### {model_name}")
            sub = navigation_df[navigation_df["model"] == model_name]
            for metric in [
                "pca_dim_95",
                "ridge_z_to_velocity_r2",
                "cca_top1_z_vs_velocity_speed",
                "best_high_error_roc_auc",
                "best_score_reject10_high_error_recall",
                "best_score_reject10_retained_rmse",
                "mean_latent_smoothness",
            ]:
                row = sub[sub["metric"] == metric]
                if len(row):
                    val = row["value"].iloc[0]
                    if np.isfinite(val):
                        lines.append(f"- `{metric}`: `{val:.4f}`")
            lines.append("")
        lines.append(
            "Interpretation: these metrics assess whether the latent can serve as a compact navigation state, "
            "a lightweight readout interface, and a risk monitor for fallback/rejection policies. They do not "
            "turn the network into a formally calibrated estimator."
        )
    else:
        lines.append("- Navigation readiness summary unavailable.")
    lines.append("")

    gains = strongest_gain_table(slice_df, with_name, without_name)
    lines += ["## Strongest Robustness Gains", ""]
    if len(gains):
        for _, row in gains.head(5).iterrows():
            lines.append(
                f"- `{row['slice']}`: delta `{row['rmse_delta_with_minus_without']:.4f}`, "
                f"relative gain `{row['relative_gain_percent']:.2f}%`"
            )
    else:
        lines.append("- Slice comparison unavailable.")
    lines.append("")

    lines += ["## Reject-option Reliability", ""]
    if len(reject_df):
        for model_name in reject_df["model"].drop_duplicates():
            sub = reject_df[(reject_df["model"] == model_name) & (reject_df["score"] == "mahalanobis")]
            if len(sub):
                r0 = sub[sub["reject_percent"] == 0]["retained_rmse"].iloc[0]
                r10 = sub[sub["reject_percent"] == 10]["retained_rmse"].iloc[0]
                lines.append(f"- `{model_name}` Mahalanobis reject 10%: RMSE `{r0:.4f}` -> `{r10:.4f}`")
    else:
        lines.append("- Reject-option table unavailable.")
    lines.append("")

    lines += ["## Operational Gate / Fallback Policy", ""]
    if len(gate_df):
        for model_name in gate_df["model"].drop_duplicates():
            sub = gate_df[(gate_df["model"] == model_name) & (gate_df["reject_percent"] == 10)]
            if len(sub):
                best = sub.sort_values("high_error_recall_rejected", ascending=False).iloc[0]
                lines.append(
                    f"- `{model_name}` best 10% gate uses `{best['score']}`: "
                    f"high-error recall `{best['high_error_recall_rejected']:.3f}`, "
                    f"rejection precision `{best['rejection_precision_high_error']:.3f}`, "
                    f"retained RMSE `{best['retained_rmse']:.4f}`."
                )
        lines.append(
            "- Use this table to choose when a navigation stack should fall back to a classical estimator, "
            "increase uncertainty, or request conservative control."
        )
    else:
        lines.append("- Operational gate table unavailable.")
    lines.append("")

    lines += ["## High-error Detection", ""]
    if len(high_df):
        best = high_df.sort_values("roc_auc", ascending=False).groupby("model").head(3)
        for _, row in best.iterrows():
            lines.append(
                f"- `{row['model']}` `{row['score']}`: ROC AUC `{row['roc_auc']:.3f}`, "
                f"PR AUC `{row['pr_auc']:.3f}`, Spearman `{row['spearman_error']:.3f}`"
            )
    else:
        lines.append("- High-error detection unavailable.")
    lines.append("")

    lines += ["## Attention Correlations", ""]
    if len(attention_df):
        top_att = attention_df.reindex(attention_df["spearman"].abs().sort_values(ascending=False).index).head(8)
        for _, row in top_att.iterrows():
            lines.append(
                f"- `{row['model']}` attention `{row['attention_group']}` vs `{row['context']}`: "
                f"Spearman `{row['spearman']:.3f}`"
            )
    else:
        lines.append("- Attention arrays missing or incompatible; attention analysis skipped.")
    lines.append("")

    lines += ["## Latent Geometry And Physical Alignment", ""]
    if len(physical_df):
        probe = physical_df[
            (physical_df["analysis"] == "ridge_probe")
            & (physical_df["target"] == "velocity")
            & (physical_df["metric"] == "r2")
        ]
        for _, row in probe.iterrows():
            lines.append(f"- `{row['model']}` latent-to-velocity ridge probe R2: `{row['value']:.4f}`")
    lines.append("- PCA/UMAP figures are saved under `figures/`.")
    lines.append("")

    lines += ["## Temporal Diagnostics", ""]
    if len(temporal_df):
        lines.append("- Temporal-contiguous diagnostics were computed from sequence IDs and timestamps.")
    else:
        lines.append(
            "- Temporal diagnostics were skipped or empty. Sequence IDs and ordered timestamps are required "
            "for claims about latent trajectory smoothness, transition probes, and residual autocorrelation."
        )
    lines.append("")

    lines += [
        "## Scientific Claim Checklist",
        "",
        "1. Flow self-supervision improves final velocity RMSE, especially in hard regimes: see `basic_performance` and `robustness_slices`.",
        "2. Structured latent space can be used pragmatically in navigation: see `navigation_readiness_summary` and `navigation_gate_policy`.",
        "3. Flow supervision reshapes latent geometry: see PCA/UMAP and physical-alignment outputs.",
        "4. Latent distances as reliability indicators: see `reject_option`, `high_error_detection`, and `navigation_gate_policy`.",
        "5. Context-dependent modality reliance: see `attention_correlations` and attention-vs-context plots.",
        "6. State-estimator-like behavior: see physical alignment and temporal diagnostics; avoid formal uncertainty claims.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global SHOW_FIGURE_TITLES
    parser = argparse.ArgumentParser(description="Journal latent/reliability analysis for ELOPE models.")
    parser.add_argument("--with-flow-dir", required=True, type=Path)
    parser.add_argument("--without-flow-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--make-umap", action="store_true")
    parser.add_argument("--temporal-only", action="store_true")
    parser.add_argument("--font-size", type=float, default=10, help="Base font size for generated figures")
    parser.add_argument("--no-figure-titles", action="store_true", help="Omit figure-level titles intended to be supplied by manuscript captions")
    args = parser.parse_args()
    SHOW_FIGURE_TITLES = not args.no_figure_titles

    setup_style(args.font_size)
    np.random.seed(args.random_seed)
    dirs = ensure_dirs(args.out_dir)

    with_flow = load_outputs(args.with_flow_dir, "with_flow_best")
    without_flow = load_outputs(args.without_flow_dir, "without_flow_best")
    with_flow, without_flow, align_mode = align_models(with_flow, without_flow)
    models = [with_flow, without_flow]

    if len(with_flow.z) == 0:
        raise ValueError("Alignment produced zero samples.")
    if not np.allclose(with_flow.true, without_flow.true, equal_nan=True):
        diff = np.linalg.norm(with_flow.true - without_flow.true, axis=1)
        mismatch = int(np.sum(diff > 1e-6))
        warnings.warn(
            "Aligned ground-truth velocity arrays differ between models; using each model's own targets. "
            f"Mismatched samples: {mismatch}/{len(diff)}, max target delta: {float(np.nanmax(diff)):.6g}. "
            "Check `arrays/run_metadata.json` for source files."
        )

    basic_df = compute_errors(with_flow, without_flow, dirs["tables"])
    slice_df = compute_slices(models, dirs["tables"], dirs["figures"])
    reject_df = reject_option_analysis(models, dirs["tables"], dirs["figures"], dirs["arrays"])
    high_df = high_error_detection(
        models,
        args.high_error_quantile,
        args.n_bootstrap,
        args.random_seed,
        dirs["tables"],
        dirs["figures"],
    )
    reliability_df = reliability_curves(models, dirs["tables"], dirs["figures"])
    attention_df = attention_analysis(models, dirs["tables"], dirs["figures"])
    physical_df = physical_alignment(models, dirs["tables"], dirs["figures"], args.random_seed)
    temporal_df = temporal_diagnostics(
        models, dirs["tables"], dirs["figures"], args.random_seed, args.temporal_only
    )
    gate_df = navigation_gate_analysis(
        models, args.high_error_quantile, dirs["tables"], dirs["figures"]
    )
    navigation_df = navigation_readiness_summary(
        models, basic_df, high_df, physical_df, temporal_df, gate_df, dirs["tables"]
    )
    manifold_plots(models, dirs["figures"], dirs["arrays"], args.make_umap, args.random_seed)
    curated_paper_plots(
        models,
        basic_df,
        slice_df,
        gate_df,
        reliability_df,
        navigation_df,
        dirs["figures_paper"],
    )

    metadata = {
        "with_flow_dir": str(args.with_flow_dir),
        "without_flow_dir": str(args.without_flow_dir),
        "out_dir": str(args.out_dir),
        "aligned_samples": len(with_flow.z),
        "alignment_mode": align_mode,
        "high_error_quantile": args.high_error_quantile,
        "n_bootstrap": args.n_bootstrap,
        "random_seed": args.random_seed,
        "make_umap": args.make_umap,
        "temporal_only": args.temporal_only,
        "with_flow_source_files": list(with_flow.source_files),
        "without_flow_source_files": list(without_flow.source_files),
    }
    (dirs["arrays"] / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report(
        dirs["root"],
        basic_df,
        slice_df,
        reject_df,
        high_df,
        attention_df,
        physical_df,
        temporal_df,
        gate_df,
        navigation_df,
        align_mode,
        models,
    )
    print(f"Saved journal latent analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()
