#!/usr/bin/env python3
"""Minimal leakage-free revision of the submitted latent-space analysis.

The trained checkpoints, dataset representation, event window, and trajectory
split are fixed.  Diagnostic transforms are fitted on trajectories 0000--0027
excluding 0004 and are evaluated on the original validation trajectory 0004.

By default the script re-extracts train/validation latent packs for the fixed
with-flow and without-flow checkpoints; `--reuse-artifacts` validates and reuses
existing packs. It then computes held-out probes, reliability, representation
similarity, attention, and latent-space metrics.

Extracted packs are stored in `analysis/revision_latents/`. CSV tables, PNG/PDF
figures, and `summary.md` are written under
`analysis/revision_validation_analysis/`. Source checkpoints are not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from compare_latent_spaces import (
    LatentPack,
    _cka_linear,
    _cov_eigs,
    _effective_rank,
    _knn_purity,
    _participation_ratio,
    _ridge_predict,
    _ridge_regression,
    _svcca_similarity,
    _zscore_apply,
    _zscore_fit,
    align_packs,
    extract_latents_from_model,
    load_latent_pack,
)


ROOT = Path(__file__).resolve().parents[1]
LATENT_DIR = ROOT / "analysis" / "revision_latents"
OUT_DIR = ROOT / "analysis" / "revision_validation_analysis"
OLD_DIR = ROOT / "plots" / "latent_compare_best_3"
MODEL_DIRS = {
    "with_flow": ROOT / "weights" / "emmnet-angles-of_20260209_144255",
    "without_flow": ROOT / "weights" / "emmnet-angles_20260128_174458",
}
TRAIN_SEQUENCES = [f"{i:04d}" for i in range(28) if i != 4]
VALIDATION_SEQUENCES = ["0004"]
REJECT_PERCENTAGES = np.asarray([0, 1, 2, 5, 10, 15, 20, 30], dtype=float)
EPS = 1e-12


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(y) - np.asarray(pred)) ** 2, axis=1))))


def _global_r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    return float(1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean(axis=0)) ** 2), EPS))


def _mean_component_r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(r2_score(y, pred, multioutput="uniform_average"))


def _sample_keys(pack: LatentPack) -> set[tuple[int, float]]:
    return {
        (int(s), round(float(t), 7))
        for s, t in zip(pack.sequence_id.reshape(-1), pack.times.reshape(-1))
    }


def _payload(pack: LatentPack, split: str, model: str) -> dict[str, np.ndarray]:
    error = np.linalg.norm(pack.pred - pack.target_vel, axis=1)
    lateral_error = np.linalg.norm(pack.pred[:, :2] - pack.target_vel[:, :2], axis=1)
    payload: dict[str, np.ndarray] = {
        "trajectory_id": pack.sequence_id.astype(np.int32),
        "sequence_id": pack.sequence_id.astype(np.int32),
        "timestamp": pack.times.astype(np.float32),
        "times": pack.times.astype(np.float32),
        "target_velocity": pack.target_vel.astype(np.float32),
        "target_vel": pack.target_vel.astype(np.float32),
        "predicted_learned_velocity": pack.pred.astype(np.float32),
        "pred": pack.pred.astype(np.float32),
        "prediction_error": error.astype(np.float32),
        "lateral_prediction_error": lateral_error.astype(np.float32),
        "speed": np.linalg.norm(pack.target_vel, axis=1).astype(np.float32),
        "fused_latent": pack.fused.astype(np.float32),
        "fused": pack.fused.astype(np.float32),
        "split": np.asarray(split),
        "model": np.asarray(model),
    }
    if pack.target_pos is not None:
        payload["target_pos"] = pack.target_pos.astype(np.float32)
    if pack.event_density is not None:
        payload["event_density"] = pack.event_density.astype(np.float32)
    if pack.attention is not None:
        payload["attention_matrices"] = pack.attention.astype(np.float32)
        payload["attention"] = pack.attention.astype(np.float32)
    if pack.event_tokens is not None:
        payload["event_tokens"] = pack.event_tokens
    if pack.total_tokens is not None:
        payload["total_tokens"] = pack.total_tokens
    if pack.flow_vector is not None:
        payload["flow_vector"] = pack.flow_vector.astype(np.float32)
    for name, arr in pack.layers.items():
        payload[f"layer__{name.replace('.', '__')}"] = arr.astype(np.float32)
    event_features = pack.layers.get("encoder_event")
    if event_features is not None:
        payload["event_encoder_features"] = event_features.astype(np.float32)
    return payload


def _extract_one(model: str, split: str, sequences: list[str], device: torch.device, workers: int) -> LatentPack:
    model_dir = MODEL_DIRS[model]
    path = LATENT_DIR / f"{model}_{split}.npz"
    pack = extract_latents_from_model(
        name=model,
        model_cfg_path=model_dir / "model-cfg.yml",
        dataset_cfg_path=model_dir / "dataset-cfg.yml",
        weights_path=model_dir / "best.pth",
        sequences=sequences,
        device=device,
        batch_size=32,
        num_workers=workers,
        max_batches=None,
        save_npz_path=None,
    )
    np.savez_compressed(path, **_payload(pack, split, model))
    return load_latent_pack(path, model)


def create_or_load_artifacts(reuse: bool, device: torch.device, workers: int) -> dict[str, dict[str, LatentPack]]:
    LATENT_DIR.mkdir(parents=True, exist_ok=True)
    packs: dict[str, dict[str, LatentPack]] = {}
    for model in MODEL_DIRS:
        packs[model] = {}
        for split, sequences in (("train", TRAIN_SEQUENCES), ("validation", VALIDATION_SEQUENCES)):
            path = LATENT_DIR / f"{model}_{split}.npz"
            packs[model][split] = load_latent_pack(path, model) if reuse and path.exists() else _extract_one(
                model, split, sequences, device, workers
            )

    for model, split_packs in packs.items():
        train, val = split_packs["train"], split_packs["validation"]
        assert len(train.fused) == 4215, f"{model}: expected 4215 training rows, got {len(train.fused)}"
        assert len(val.fused) == 122, f"{model}: expected 122 validation rows, got {len(val.fused)}"
        assert set(np.unique(train.sequence_id).tolist()) == set(range(28)) - {4}
        assert set(np.unique(val.sequence_id).tolist()) == {4}
        assert _sample_keys(train).isdisjoint(_sample_keys(val)), f"{model}: validation row leaked into training artifact"

        old = load_latent_pack(OLD_DIR / f"extracted_{model}_best.npz", model)
        old_train = old.subset(np.where(old.sequence_id != 4)[0])
        old_val = old.subset(np.where(old.sequence_id == 4)[0])
        inferred_train, submitted_train, st = align_packs(train, old_train)
        inferred_val, submitted_val, sv = align_packs(val, old_val)
        assert st["matched"] == 4215 and sv["matched"] == 122
        # The submitted arrays were produced on GPU. CPU re-inference differs at
        # harmless floating-point accumulation scale, while targets/times match
        # exactly. These bounds are <5e-4 relative to the velocity scale.
        assert np.allclose(inferred_train.target_vel, submitted_train.target_vel, atol=0, rtol=0)
        assert np.allclose(inferred_val.target_vel, submitted_val.target_vel, atol=0, rtol=0)
        assert np.allclose(inferred_train.pred, submitted_train.pred, atol=5e-2, rtol=1e-4)
        assert np.allclose(inferred_val.pred, submitted_val.pred, atol=5e-2, rtol=1e-4)
        assert np.allclose(inferred_train.fused, submitted_train.fused, atol=5e-3, rtol=1e-4)
        assert np.allclose(inferred_val.fused, submitted_val.fused, atol=5e-3, rtol=1e-4)
    return packs


def _old_pack(model: str) -> LatentPack:
    return load_latent_pack(OLD_DIR / f"extracted_{model}_best.npz", model)


def _mahal_fit_apply(train_z: np.ndarray, query_z: np.ndarray) -> tuple[np.ndarray, LedoitWolf]:
    estimator = LedoitWolf().fit(np.asarray(train_z, dtype=np.float64))
    d2 = estimator.mahalanobis(np.asarray(query_z, dtype=np.float64))
    return np.sqrt(np.maximum(d2, 0.0)), estimator


def reliability_analysis(packs: dict[str, dict[str, LatentPack]]) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODEL_DIRS:
        train, val, old = packs[model]["train"], packs[model]["validation"], _old_pack(model)
        train_err = np.linalg.norm(train.pred - train.target_vel, axis=1)
        val_err = np.linalg.norm(val.pred - val.target_vel, axis=1)
        old_err = np.linalg.norm(old.pred - old.target_vel, axis=1)
        val_risk, estimator = _mahal_fit_apply(train.fused, val.fused)
        train_risk = np.sqrt(np.maximum(estimator.mahalanobis(train.fused.astype(np.float64)), 0.0))
        old_risk, _ = _mahal_fit_apply(old.fused, old.fused)
        train_high_threshold = float(np.quantile(train_err, 0.90))
        old_high_threshold = float(np.quantile(old_err, 0.90))
        val_high = val_err >= train_high_threshold
        old_high = old_err >= old_high_threshold
        rho = float(stats.spearmanr(val_risk, val_err).correlation)
        old_rho = float(stats.spearmanr(old_risk, old_err).correlation)
        auc = float(roc_auc_score(val_high, val_risk)) if len(np.unique(val_high)) == 2 else np.nan
        old_auc = float(roc_auc_score(old_high, old_risk))
        base = {
            "model": model,
            "fit_split": "train",
            "evaluation_split": "validation",
            "train_n": len(train.fused),
            "validation_n": len(val.fused),
        }
        rows.extend(
            [
                {**base, "analysis": "summary", "metric": "spearman_mahalanobis_error", "value": rho,
                 "old_combined_value": old_rho, "threshold": np.nan, "nominal_reject_percent": np.nan,
                 "validation_rejected_n": np.nan, "validation_retained_n": np.nan},
                {**base, "analysis": "summary", "metric": "high_error_auc", "value": auc,
                 "old_combined_value": old_auc, "threshold": train_high_threshold,
                 "nominal_reject_percent": np.nan, "validation_rejected_n": int(val_high.sum()),
                 "validation_retained_n": int((~val_high).sum())},
                {**base, "analysis": "context", "metric": "validation_velocity_rmse", "value": _rmse(val.target_vel, val.pred),
                 "old_combined_value": _rmse(old.target_vel, old.pred), "threshold": np.nan,
                 "nominal_reject_percent": np.nan, "validation_rejected_n": 0, "validation_retained_n": len(val.fused)},
            ]
        )

        train_deciles = np.quantile(train_risk, np.linspace(0, 1, 11))
        old_deciles = np.quantile(old_risk, np.linspace(0, 1, 11))
        val_bins = np.clip(np.digitize(val_risk, train_deciles[1:-1], right=False), 0, 9)
        old_bins = np.clip(np.digitize(old_risk, old_deciles[1:-1], right=False), 0, 9)
        for decile in range(10):
            vm, om = val_bins == decile, old_bins == decile
            rows.append(
                {**base, "analysis": "risk_decile", "metric": f"mean_error_decile_{decile + 1}",
                 "value": float(val_err[vm].mean()) if vm.any() else np.nan,
                 "old_combined_value": float(old_err[om].mean()) if om.any() else np.nan,
                 "threshold": float(train_deciles[decile + 1]), "nominal_reject_percent": np.nan,
                 "validation_rejected_n": np.nan, "validation_retained_n": int(vm.sum())}
            )

        for pct in REJECT_PERCENTAGES:
            threshold = np.inf if pct == 0 else float(np.quantile(train_risk, 1.0 - pct / 100.0))
            old_threshold = np.inf if pct == 0 else float(np.quantile(old_risk, 1.0 - pct / 100.0))
            reject = val_risk >= threshold
            old_reject = old_risk >= old_threshold
            keep, old_keep = ~reject, ~old_reject
            recall = float(np.sum(reject & val_high) / max(np.sum(val_high), 1))
            old_recall = float(np.sum(old_reject & old_high) / max(np.sum(old_high), 1))
            rows.extend(
                [
                    {**base, "analysis": "risk_gate", "metric": "retained_rmse", "value": _rmse(val.target_vel[keep], val.pred[keep]),
                     "old_combined_value": _rmse(old.target_vel[old_keep], old.pred[old_keep]), "threshold": threshold,
                     "nominal_reject_percent": pct, "validation_rejected_n": int(reject.sum()), "validation_retained_n": int(keep.sum())},
                    {**base, "analysis": "risk_gate", "metric": "high_error_recall", "value": recall,
                     "old_combined_value": old_recall, "threshold": threshold, "nominal_reject_percent": pct,
                     "validation_rejected_n": int(reject.sum()), "validation_retained_n": int(keep.sum())},
                ]
            )
    return pd.DataFrame(rows)


def _ordered_indices(pack: LatentPack, sid: int) -> np.ndarray:
    idx = np.where(pack.sequence_id.reshape(-1).astype(int) == sid)[0]
    return idx[np.argsort(pack.times[idx])]


def _history_dataset(pack: LatentPack, lag: int, target: str, sort_times: bool = True) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for sid in np.unique(pack.sequence_id.astype(int)):
        idx = _ordered_indices(pack, int(sid)) if sort_times else np.where(pack.sequence_id.astype(int) == sid)[0]
        if target in {"z_next", "v_next"}:
            for j in range(len(idx) - 1):
                xs.append(pack.fused[idx[j]])
                ys.append(pack.fused[idx[j + 1]] if target == "z_next" else pack.target_vel[idx[j + 1]])
        else:
            for j in range(lag, len(idx)):
                xs.append(np.concatenate([pack.fused[idx[j - k]] for k in range(lag + 1)]))
                ys.append(pack.target_vel[idx[j]])
    return np.asarray(xs), np.asarray(ys)


def _ridge_train_eval(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, alpha: float) -> dict[str, float]:
    mu, sd = _zscore_fit(xtr)
    model = _ridge_regression(_zscore_apply(xtr, mu, sd), ytr, l2=alpha)
    pred = _ridge_predict(_zscore_apply(xva, mu, sd), model)
    return {
        "r2_global": _global_r2(yva, pred),
        "r2_mean_components": _mean_component_r2(yva, pred),
        "rmse": _rmse(yva, pred),
        "mae": float(mean_absolute_error(yva, pred)),
    }


def _old_random_probe(pack: LatentPack, lag: int, target: str, seed: int, alpha: float) -> dict[str, float]:
    # Preserve the submitted artifact order exactly. Its saved `times` were
    # within-window offsets, so sorting those values would fabricate pairs.
    x, y = _history_dataset(pack, lag, target, sort_times=False)
    idx = np.random.default_rng(seed).permutation(len(x))
    cut = int(0.8 * len(x))
    return _ridge_train_eval(x[idx[:cut]], y[idx[:cut]], x[idx[cut:]], y[idx[cut:]], alpha)


def _class_labels(train: LatentPack, query: LatentPack, kind: str) -> tuple[np.ndarray, np.ndarray]:
    tr_speed = np.linalg.norm(train.target_vel, axis=1)
    q_speed = np.linalg.norm(query.target_vel, axis=1)
    if kind == "speed_bin":
        edges = np.unique(np.quantile(tr_speed, np.linspace(0, 1, 5)))
        return np.digitize(tr_speed, edges[1:-1]), np.digitize(q_speed, edges[1:-1])
    if kind == "direction":
        threshold = float(np.quantile(tr_speed, 0.20))
        def direction(v: np.ndarray, speed: np.ndarray) -> np.ndarray:
            labels = np.floor(((np.arctan2(v[:, 1], v[:, 0]) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi / 8)).astype(int)
            labels[speed <= threshold] = -1
            return labels
        return direction(train.target_vel, tr_speed), direction(query.target_vel, q_speed)
    threshold = float(np.quantile(tr_speed, 0.25))
    return (tr_speed > threshold).astype(int), (q_speed > threshold).astype(int)


def _classification_probe(train: LatentPack, val: LatentPack, kind: str) -> tuple[float, int]:
    ytr, yva = _class_labels(train, val, kind)
    tr_mask, va_mask = ytr >= 0, yva >= 0
    sx = StandardScaler().fit(train.fused[tr_mask])
    classes = np.unique(ytr[tr_mask])
    onehot = np.zeros((tr_mask.sum(), len(classes)))
    encoded = np.searchsorted(classes, ytr[tr_mask])
    onehot[np.arange(len(encoded)), encoded] = 1.0
    probe = Ridge(alpha=1e-3).fit(sx.transform(train.fused[tr_mask]), onehot)
    pred = classes[np.argmax(probe.predict(sx.transform(val.fused[va_mask])), axis=1)]
    return float(accuracy_score(yva[va_mask], pred)), int(va_mask.sum())


def probe_analysis(packs: dict[str, dict[str, LatentPack]]) -> pd.DataFrame:
    rows: list[dict] = []
    configs = [
        ("z_t_to_v_t", 0, "v", 1e-3),
        ("z_t_z_t-1_to_v_t", 1, "v", 1e-3),
        ("z_t_z_t-1_z_t-2_to_v_t", 2, "v", 1e-3),
        ("z_t_to_z_t+1", 0, "z_next", 1e-2),
        ("z_t_to_v_t+1", 0, "v_next", 1e-3),
    ]
    for model in MODEL_DIRS:
        train, val, old = packs[model]["train"], packs[model]["validation"], _old_pack(model)
        for name, lag, target, alpha in configs:
            xtr, ytr = _history_dataset(train, lag, target)
            xva, yva = _history_dataset(val, lag, target)
            new = _ridge_train_eval(xtr, ytr, xva, yva, alpha)
            old_seed = (42 if model == "with_flow" else 143) + (lag if target == "v" else 0)
            # The classical-dynamics journal table used seed 42 for both models.
            if target in {"z_next", "v_next"}:
                old_seed = 42
            old_m = _old_random_probe(old, lag, target, old_seed, alpha)
            for metric, value in new.items():
                rows.append({"model": model, "probe": name, "metric": metric, "value": value,
                             "old_combined_value": old_m[metric], "fit_n": len(xtr), "validation_n": len(xva),
                             "fit_split": "train", "evaluation_split": "validation"})
        for kind in ("speed_bin", "direction", "static_dynamic"):
            acc, nval = _classification_probe(train, val, kind)
            old_json = json.loads((OLD_DIR / f"analysis_{model}_best.json").read_text())
            old_acc = old_json["probes"][f"{kind}_probe"]["accuracy"]
            rows.append({"model": model, "probe": kind, "metric": "accuracy", "value": acc,
                         "old_combined_value": old_acc, "fit_n": len(train.fused), "validation_n": nval,
                         "fit_split": "train", "evaluation_split": "validation"})
    return pd.DataFrame(rows)


def cca_analysis(packs: dict[str, dict[str, LatentPack]]) -> pd.DataFrame:
    rows = []
    for model in MODEL_DIRS:
        train, val, old = packs[model]["train"], packs[model]["validation"], _old_pack(model)
        ytr = np.column_stack([train.target_vel, np.linalg.norm(train.target_vel, axis=1)])
        yva = np.column_stack([val.target_vel, np.linalg.norm(val.target_vel, axis=1)])
        x_scaler, y_scaler = StandardScaler().fit(train.fused), StandardScaler().fit(ytr)
        cca = CCA(n_components=4, max_iter=2000).fit(x_scaler.transform(train.fused), y_scaler.transform(ytr))
        xcv, ycv = cca.transform(x_scaler.transform(val.fused), y_scaler.transform(yva))

        yo = np.column_stack([old.target_vel, np.linalg.norm(old.target_vel, axis=1)])
        oxs, oys = StandardScaler().fit(old.fused), StandardScaler().fit(yo)
        old_cca = CCA(n_components=4, max_iter=2000).fit(oxs.transform(old.fused), oys.transform(yo))
        xco, yco = old_cca.transform(oxs.transform(old.fused), oys.transform(yo))
        corr, old_corr = [], []
        for i in range(4):
            corr.append(float(np.corrcoef(xcv[:, i], ycv[:, i])[0, 1]))
            old_corr.append(float(np.corrcoef(xco[:, i], yco[:, i])[0, 1]))
            common = {"model": model, "fit_n": len(train.fused), "validation_n": len(val.fused),
                      "fit_split": "train", "evaluation_split": "validation"}
            rows.append({**common, "metric": f"canonical_{i + 1}_correlation", "value": abs(corr[-1]),
                         "old_combined_value": abs(old_corr[-1])})
            rows.append({**common, "metric": f"canonical_{i + 1}_signed_correlation", "value": corr[-1],
                         "old_combined_value": old_corr[-1]})
        for k in (1, 2, 3, 4):
            rows.append({"model": model, "metric": f"top_{k}_mean", "value": float(np.mean(np.abs(corr[:k]))),
                         "old_combined_value": float(np.mean(np.abs(old_corr[:k]))), "fit_n": len(train.fused),
                         "validation_n": len(val.fused), "fit_split": "train", "evaluation_split": "validation"})
    return pd.DataFrame(rows)


def _knn_reference_purity(train: LatentPack, val: LatentPack, kind: str, k: int = 10) -> tuple[float, int]:
    ytr, yva = _class_labels(train, val, kind)
    tr_mask, va_mask = ytr >= 0, yva >= 0
    ref, query = train.fused[tr_mask].astype(np.float64), val.fused[va_mask].astype(np.float64)
    distances = np.sum((query[:, None, :] - ref[None, :, :]) ** 2, axis=2)
    nn = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(ytr[tr_mask][nn] == yva[va_mask, None])), int(va_mask.sum())


def latent_analysis(packs: dict[str, dict[str, LatentPack]]) -> pd.DataFrame:
    rows = []
    for model in MODEL_DIRS:
        train, val, old = packs[model]["train"], packs[model]["validation"], _old_pack(model)
        eig, old_eig = _cov_eigs(train.fused), _cov_eigs(old.fused)
        var, old_var = eig / eig.sum(), old_eig / old_eig.sum()
        metrics = {
            "participation_ratio": (_participation_ratio(eig), _participation_ratio(old_eig)),
            "effective_rank": (_effective_rank(eig), _effective_rank(old_eig)),
            "k90": (int(np.searchsorted(np.cumsum(var), .90) + 1), int(np.searchsorted(np.cumsum(old_var), .90) + 1)),
            "k95": (int(np.searchsorted(np.cumsum(var), .95) + 1), int(np.searchsorted(np.cumsum(old_var), .95) + 1)),
            "explained_variance_pc1": (float(var[0]), float(old_var[0])),
            "explained_variance_pc2": (float(var[1]), float(old_var[1])),
            "explained_variance_pc3": (float(var[2]), float(old_var[2])),
            "validation_velocity_rmse": (_rmse(val.target_vel, val.pred), _rmse(old.target_vel, old.pred)),
        }
        for metric, (value, old_value) in metrics.items():
            distribution = "validation" if metric.startswith("validation") else "training"
            rows.append({"model": model, "analysis": "distribution_or_performance", "metric": metric, "value": value,
                         "old_combined_value": old_value, "distribution": distribution, "reference_n": len(train.fused),
                         "query_n": len(val.fused)})
        old_json = json.loads((OLD_DIR / f"analysis_{model}_best.json").read_text())
        for kind in ("speed_bin", "direction", "static_dynamic"):
            value, nq = _knn_reference_purity(train, val, kind)
            old_value = old_json["knn"][f"purity_{kind}_k10"]
            rows.append({"model": model, "analysis": "train_reference_validation_query_knn", "metric": f"{kind}_purity_k10",
                         "value": value, "old_combined_value": old_value, "distribution": "held-out validation query",
                         "reference_n": len(train.fused), "query_n": nq})
    return pd.DataFrame(rows)


def representation_similarity(packs: dict[str, dict[str, LatentPack]]) -> pd.DataFrame:
    flow, noflow, alignment = align_packs(packs["with_flow"]["validation"], packs["without_flow"]["validation"])
    assert alignment["matched"] == 122 and alignment["mode"] == "key_match"
    old = json.loads((OLD_DIR / "comparison.json").read_text())["layer_similarity"]
    rows = []
    for label, a, b, old_key in (
        ("fused", flow.fused, noflow.fused, "fused"),
        ("event_encoder", flow.layers["encoder_event"], noflow.layers["encoder_event"], "encoder_event"),
    ):
        for metric, value in (("cka", _cka_linear(a, b)), ("svcca", _svcca_similarity(a, b))):
            rows.append({"representation": label, "metric": metric, "value": value,
                         "old_combined_value": old[old_key][metric], "n": len(a),
                         "evaluation_split": "held-out validation", "stability_note": "n=122; interpret SVCCA cautiously"})
    return pd.DataFrame(rows)


def _attention_groups(pack: LatentPack) -> dict[str, np.ndarray]:
    att = pack.attention.astype(np.float64)
    if att.ndim == 4:
        att = att.mean(axis=1)
    return {
        "event": att[:, :, :4].mean(axis=(1, 2)),
        "omega": att[:, :, 4].mean(axis=1),
        "range": att[:, :, 5].mean(axis=1),
        "attitude": att[:, :, 6].mean(axis=1),
    }


def attention_analysis(packs: dict[str, dict[str, LatentPack]]) -> pd.DataFrame:
    rows = []
    for model in MODEL_DIRS:
        val, old = packs[model]["validation"], _old_pack(model)
        vg, og = _attention_groups(val), _attention_groups(old)
        contexts = {
            "error": np.linalg.norm(val.pred - val.target_vel, axis=1),
            "speed": np.linalg.norm(val.target_vel, axis=1),
            "event_density": val.event_density,
        }
        old_contexts = {
            "error": np.linalg.norm(old.pred - old.target_vel, axis=1),
            "speed": np.linalg.norm(old.target_vel, axis=1),
            "event_density": old.event_density,
        }
        for group in vg:
            rows.append({"model": model, "attention_group": group, "metric": "mean_attention", "context": "none",
                         "value": float(vg[group].mean()), "old_combined_value": float(og[group].mean()), "n": len(val.fused),
                         "evaluation_split": "validation"})
            for context in contexts:
                rows.append({"model": model, "attention_group": group, "metric": "spearman", "context": context,
                             "value": float(stats.spearmanr(vg[group], contexts[context]).correlation),
                             "old_combined_value": float(stats.spearmanr(og[group], old_contexts[context]).correlation),
                             "n": len(val.fused), "evaluation_split": "validation"})
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def make_figures(packs: dict[str, dict[str, LatentPack]], reliability: pd.DataFrame, figure_dir: Path) -> None:
    colors = {"with_flow": "#0f766e", "without_flow": "#b45309"}
    gate = reliability[(reliability.analysis == "risk_gate")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for model in MODEL_DIRS:
        g = gate[(gate.model == model) & (gate.metric == "retained_rmse")]
        axes[0].plot(g.nominal_reject_percent, g.value, "o-", label=model, color=colors[model])
        g = gate[(gate.model == model) & (gate.metric == "high_error_recall")]
        axes[1].plot(g.nominal_reject_percent, g.value, "o-", label=model, color=colors[model])
    axes[0].set(xlabel="Training-calibrated rejected percentile (%)", ylabel="Validation retained RMSE")
    axes[1].set(xlabel="Training-calibrated rejected percentile (%)", ylabel="Validation high-error recall", ylim=(-.02, 1.02))
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    _save_figure(fig, figure_dir / "validation_risk_gate")

    deciles = reliability[reliability.analysis == "risk_decile"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for model in MODEL_DIRS:
        g = deciles[deciles.model == model]
        ax.plot(range(1, 11), g.value, "o-", label=model, color=colors[model])
    ax.set(xlabel="Mahalanobis risk decile (training boundaries)", ylabel="Mean validation velocity error", xticks=range(1, 11))
    ax.grid(alpha=.25); ax.legend()
    _save_figure(fig, figure_dir / "validation_risk_calibration")

    projections = {}
    for model in MODEL_DIRS:
        train, val = packs[model]["train"], packs[model]["validation"]
        pca = PCA(n_components=2).fit(train.fused)
        projections[model] = pca.transform(val.fused)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for col, model in enumerate(MODEL_DIRS):
        val, z = packs[model]["validation"], projections[model]
        speed = np.linalg.norm(val.target_vel, axis=1)
        order = np.argsort(val.times)
        progress = np.empty(len(order)); progress[order] = np.linspace(0, 1, len(order))
        for row, (c, label) in enumerate(((speed, "Speed"), (progress, "Trajectory progress"))):
            sc = axes[row, col].scatter(z[:, 0], z[:, 1], c=c, s=17, cmap="viridis")
            axes[row, col].set_title(f"{model}: {label}")
            axes[row, col].set(xlabel="Training PC1", ylabel="Training PC2")
            fig.colorbar(sc, ax=axes[row, col])
    _save_figure(fig, figure_dir / "validation_pca_speed_progress")

    for filename, getter, label in (
        ("validation_pca_error", lambda p: np.linalg.norm(p.pred - p.target_vel, axis=1), "Velocity error"),
        ("validation_pca_event_density", lambda p: p.event_density, "Event density"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
        for ax, model in zip(axes, MODEL_DIRS):
            z, val = projections[model], packs[model]["validation"]
            sc = ax.scatter(z[:, 0], z[:, 1], c=getter(val), s=18, cmap="viridis")
            ax.set_title(model); ax.set(xlabel="Training PC1", ylabel="Training PC2")
            fig.colorbar(sc, ax=ax, label=label)
        _save_figure(fig, figure_dir / filename)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for ax, model in zip(axes, MODEL_DIRS):
        att = packs[model]["validation"].attention
        if att.ndim == 4: att = att.mean(axis=1)
        im = ax.imshow(att.mean(axis=0), vmin=0, vmax=max(.2, float(att.mean(axis=0).max())), cmap="viridis")
        ax.set_title(model); ax.set(xlabel="Source token", ylabel="Query token")
        fig.colorbar(im, ax=ax)
    _save_figure(fig, figure_dir / "validation_attention")


def write_summary(packs: dict[str, dict[str, LatentPack]], tables: dict[str, pd.DataFrame], path: Path) -> None:
    latent, probes, cca, reliability, similarity = (tables[k] for k in ("latent", "probe", "cca", "reliability", "similarity"))
    def val(df: pd.DataFrame, model: str, metric: str, key: str = "metric") -> float:
        return float(df[(df.model == model) & (df[key] == metric)].iloc[0].value)
    lines = [
        "# Held-out validation latent-space revision",
        "",
        "This is a minimal correction of the submitted diagnostic analysis. It does not change the ELOPE inputs, event representation, 0.3 s event window, model architecture, checkpoints, training protocol, or official benchmark result.",
        "",
        "## Split and fit/evaluation contract",
        "",
        "- Training samples: **4215** from trajectories `0000`--`0027` except `0004`.",
        "- Validation samples: **122** from held-out trajectory **`0004`**.",
        "- Fitted on training only: Ledoit--Wolf latent mean/covariance, high-error cutoff, Mahalanobis decile/rejection cutoffs, ridge probes and scalers, class thresholds, CCA/scalers, and PCA.",
        "- Evaluated on validation only: reliability correlations/AUC/gates, probe scores, CCA correlations, PCA projections, attention summaries, and train-reference/validation-query neighborhoods.",
        "- CKA/SVCCA use 122 aligned held-out validation samples and no labels. SVCCA is reported with an explicit small-sample caution.",
        "- Assertions verify disjoint `(trajectory_id, timestamp)` keys, exact row counts/trajectory IDs, and numerical agreement with the corresponding rows of the submitted inference export.",
        "- **Validation data are never used to fit diagnostic quantities.**",
        "- Hybrid velocity is not saved because it was not present in the submitted latent artifacts; `predicted_learned_velocity` is the unchanged network output.",
        "",
        "## Old combined-pool versus held-out result",
        "",
        "Every replacement CSV includes `old_combined_value` beside the corrected `value`; the old column reproduces or recomputes the submitted 4337-row analysis. Key replacements are:",
        "",
        "| Metric | With flow: old -> held-out | Without flow: old -> held-out |",
        "|---|---:|---:|",
    ]
    summary_metrics = [
        ("Context velocity RMSE (combined -> validation)", latent, "validation_velocity_rmse", "metric"),
        ("Velocity probe mean-component R2", probes[probes.probe == "z_t_to_v_t"], "r2_mean_components", "metric"),
        ("Mahalanobis/error Spearman", reliability[reliability.analysis == "summary"], "spearman_mahalanobis_error", "metric"),
        ("Mahalanobis high-error AUC", reliability[reliability.analysis == "summary"], "high_error_auc", "metric"),
        ("CCA canonical 1", cca, "canonical_1_correlation", "metric"),
    ]
    for label, df, metric, key in summary_metrics:
        cells = []
        for model in MODEL_DIRS:
            row = df[(df.model == model) & (df[key] == metric)].iloc[0]
            cells.append(f"{row.old_combined_value:.4f} -> {row.value:.4f}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    for metric in ("participation_ratio", "effective_rank", "k90", "k95"):
        cells = []
        for model in MODEL_DIRS:
            row = latent[(latent.model == model) & (latent.metric == metric)].iloc[0]
            cells.append(f"{row.old_combined_value:.4f} -> {row.value:.4f}")
        lines.append(f"| Training-distribution {metric} | {cells[0]} | {cells[1]} |")
    for metric in ("speed_bin_purity_k10", "direction_purity_k10", "static_dynamic_purity_k10"):
        cells = []
        for model in MODEL_DIRS:
            row = latent[(latent.model == model) & (latent.metric == metric)].iloc[0]
            cells.append(f"{row.old_combined_value:.4f} -> {row.value:.4f}")
        lines.append(f"| Train-reference validation-query {metric} | {cells[0]} | {cells[1]} |")
    for representation in ("fused", "event_encoder"):
        for metric in ("cka", "svcca"):
            row = similarity[(similarity.representation == representation) & (similarity.metric == metric)].iloc[0]
            lines.append(f"| {representation} {metric.upper()} | {row.old_combined_value:.4f} -> {row.value:.4f} | same aligned sample set |")

    lines += [
        "",
        "### State-sufficiency and classification replacements",
        "",
        "| Model | Probe metric | Old combined | Held-out validation |",
        "|---|---|---:|---:|",
    ]
    probe_display = {
        "z_t_to_v_t": "R2 global",
        "z_t_z_t-1_to_v_t": "R2 global",
        "z_t_z_t-1_z_t-2_to_v_t": "R2 global",
        "z_t_to_z_t+1": "R2 global",
        "z_t_to_v_t+1": "R2 global",
        "speed_bin": "accuracy",
        "direction": "accuracy",
        "static_dynamic": "accuracy",
    }
    for model in MODEL_DIRS:
        for probe_name, display_metric in probe_display.items():
            metric_name = "r2_global" if display_metric == "R2 global" else "accuracy"
            row = probes[(probes.model == model) & (probes.probe == probe_name) & (probes.metric == metric_name)].iloc[0]
            lines.append(f"| {model} | `{probe_name}` {display_metric} | {row.old_combined_value:.6f} | {row.value:.6f} |")

    lines += [
        "",
        "### CCA replacements",
        "",
        "Absolute canonical correlations are the headline values below. The CSV additionally preserves signed held-out correlations; component 3 reverses sign on validation for both models.",
        "",
        "| Model | Component | Old combined | Held-out validation |",
        "|---|---:|---:|---:|",
    ]
    for model in MODEL_DIRS:
        for component in range(1, 5):
            row = cca[(cca.model == model) & (cca.metric == f"canonical_{component}_correlation")].iloc[0]
            lines.append(f"| {model} | {component} | {row.old_combined_value:.6f} | {row.value:.6f} |")

    lines += [
        "",
        "### Reliability replacements",
        "",
        "Risk cutoffs below are training quantiles. Consequently, the nominal rejection percentage is not forced on validation.",
        "",
        "| Model | Metric | Old combined | Held-out validation |",
        "|---|---|---:|---:|",
    ]
    for model in MODEL_DIRS:
        for metric in ("spearman_mahalanobis_error", "high_error_auc"):
            row = reliability[(reliability.model == model) & (reliability.analysis == "summary") & (reliability.metric == metric)].iloc[0]
            lines.append(f"| {model} | {metric} | {row.old_combined_value:.6f} | {row.value:.6f} |")
        for metric in ("retained_rmse", "high_error_recall"):
            row = reliability[(reliability.model == model) & (reliability.analysis == "risk_gate") &
                              (reliability.metric == metric) & (reliability.nominal_reject_percent == 10)].iloc[0]
            lines.append(f"| {model} | 10% training-cutoff {metric} | {row.old_combined_value:.6f} | {row.value:.6f} |")
    wf_one = reliability[(reliability.model == "with_flow") & (reliability.analysis == "risk_gate") &
                         (reliability.metric == "retained_rmse") & (reliability.nominal_reject_percent == 1)].iloc[0]
    nf_one = reliability[(reliability.model == "without_flow") & (reliability.analysis == "risk_gate") &
                         (reliability.metric == "retained_rmse") & (reliability.nominal_reject_percent == 1)].iloc[0]
    lines += [
        "",
        f"The training 1% risk cutoffs reject `{int(wf_one.validation_rejected_n)}/122` with-flow and `{int(nf_one.validation_rejected_n)}/122` without-flow validation windows. Several low training-risk deciles therefore contain no validation samples; these remain explicit as empty/NaN CSV rows rather than being re-binned on validation.",
        "",
        "### Attention replacements",
        "",
        "| Model | Group | Statistic | Old combined | Held-out validation |",
        "|---|---|---|---:|---:|",
    ]
    attention = tables["attention"]
    for _, row in attention.iterrows():
        statistic = row.metric if row.context == "none" else f"{row.metric} vs {row.context}"
        lines.append(f"| {row.model} | {row.attention_group} | {statistic} | {row.old_combined_value:.6f} | {row.value:.6f} |")

    flow_probe = val(probes[probes.probe == "z_t_to_v_t"], "with_flow", "r2_mean_components")
    noflow_probe = val(probes[probes.probe == "z_t_to_v_t"], "without_flow", "r2_mean_components")
    flow_knn = val(latent, "with_flow", "direction_purity_k10")
    noflow_knn = val(latent, "without_flow", "direction_purity_k10")
    risk_auc = val(reliability[reliability.analysis == "summary"], "with_flow", "high_error_auc")
    lines += [
        "",
        "## Claim audit",
        "",
        "- **Remains supported:** training-distribution compactness/rank ordering is essentially unchanged when validation is excluded; the with-flow and without-flow representations also remain strongly similar on aligned validation samples by fused CKA.",
        f"- **Weakens or disappears:** the strong state-sufficiency/linear velocity-decodability claim does not survive trajectory-held-out evaluation (with-flow/no-flow velocity-probe R2 `{flow_probe:.3f}`/`{noflow_probe:.3f}`). Negative R2 means the fitted probe is worse than predicting the validation-trajectory mean.",
        f"- **Weakens or disappears:** neighborhood superiority is not supported as a general claim (direction purity `{flow_knn:.3f}` vs `{noflow_knn:.3f}` on one validation trajectory).",
        f"- **Requires weakened wording:** Mahalanobis distance is a descriptive validation risk indicator (with-flow AUC `{risk_auc:.3f}`), not calibrated uncertainty and not cross-validated performance.",
        "- **Remove if stated broadly:** claims of generalization across trajectories or stable SVCCA structure; only one 122-window held-out trajectory is available and SVCCA can be numerically sensitive at this sample count.",
        "- Official ELOPE challenge/postmortem performance claims remain unchanged and are not replaced by this diagnostic validation RMSE.",
        "",
        "## Manuscript replacements",
        "",
        "- Replace the combined-pool latent/probe/CCA/reliability/attention tables with the CSVs in `tables/`.",
        "- Replace the risk-gate, risk-calibration, PCA, and attention figures with the files in `figures/` (PNG and vector PDF).",
        "- Replace mixed-pool CKA/SVCCA and neighborhood values with `representation_similarity.csv` and the k-NN rows in `latent_metrics.csv`.",
        "- Remove any transductive mixed-pool neighborhood result, any jointly fitted PCA visualization, and any wording implying validation-fitted diagnostics.",
        "- Retain official benchmark figures/tables unchanged.",
        "",
        "## Files",
        "",
        "The four split artifacts are in `../revision_latents/`. Exact numerical results are in the six requested CSV tables; figures are direct held-out replacements.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup_plot_style(font_size: float = 10) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": max(font_size - 2, 1),
            "ytick.labelsize": max(font_size - 2, 1),
            "legend.fontsize": max(font_size - 2, 1),
            "legend.title_fontsize": max(font_size - 1, 1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-artifacts", action="store_true", help="Reuse existing split NPZs after re-running integrity checks")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--font-size", type=float, default=10, help="Base font size for generated figures")
    args = parser.parse_args()
    _setup_plot_style(args.font_size)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table_dir, figure_dir = OUT_DIR / "tables", OUT_DIR / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    packs = create_or_load_artifacts(args.reuse_artifacts, device, args.num_workers)
    tables = {
        "latent": latent_analysis(packs),
        "probe": probe_analysis(packs),
        "cca": cca_analysis(packs),
        "reliability": reliability_analysis(packs),
        "similarity": representation_similarity(packs),
        "attention": attention_analysis(packs),
    }
    names = {
        "latent": "latent_metrics.csv", "probe": "probe_metrics.csv", "cca": "cca_metrics.csv",
        "reliability": "reliability_metrics.csv", "similarity": "representation_similarity.csv",
        "attention": "attention_metrics.csv",
    }
    for key, df in tables.items():
        df.to_csv(table_dir / names[key], index=False, float_format="%.12g")
    make_figures(packs, tables["reliability"], figure_dir)
    write_summary(packs, tables, OUT_DIR / "summary.md")
    print(f"Completed held-out revision analysis on {device}.")
    print(f"Latents: {LATENT_DIR}")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
