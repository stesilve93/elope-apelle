import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None


EPS = 1e-12


def _setup_plot_style() -> None:
    if not HAS_MPL:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.65,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#cbd5e1",
            "font.size": 11,
        }
    )


@dataclass
class Pack:
    name: str
    z: np.ndarray
    pred: np.ndarray
    vel: np.ndarray
    pos: np.ndarray
    event_density: np.ndarray
    seq: np.ndarray


def load_pack(path: Path, name: str) -> Pack:
    data = np.load(path, allow_pickle=True)
    required = ["fused", "pred", "target_vel", "target_pos", "event_density", "sequence_id"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"{path}: missing key `{k}`")

    seq = data["sequence_id"].reshape(-1)
    if seq.dtype.kind in {"U", "S", "O"}:
        parsed = []
        for x in seq:
            sx = str(x)
            parsed.append(int(sx) if sx.isdigit() else -1)
        seq = np.asarray(parsed, dtype=np.int32)
    else:
        seq = seq.astype(np.int32)

    return Pack(
        name=name,
        z=np.asarray(data["fused"], dtype=np.float64),
        pred=np.asarray(data["pred"], dtype=np.float64),
        vel=np.asarray(data["target_vel"], dtype=np.float64),
        pos=np.asarray(data["target_pos"], dtype=np.float64),
        event_density=np.asarray(data["event_density"], dtype=np.float64).reshape(-1),
        seq=seq,
    )


def _split_idx(n: int, seed: int = 42, frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = max(1, int(frac * n))
    return idx[:n_tr], idx[n_tr:]


def _zfit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return mu, sd


def _zapply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _ridge_fit(X: np.ndarray, Y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    I = np.eye(X1.shape[1])
    I[-1, -1] = 0.0
    A = X1.T @ X1 + l2 * I
    B = X1.T @ Y
    return np.linalg.solve(A, B)


def _ridge_pred(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return X1 @ W


def _r2_global(Y: np.ndarray, Yh: np.ndarray) -> float:
    sse = np.sum((Y - Yh) ** 2)
    sst = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - sse / max(sst, EPS))


def _r2_components(Y: np.ndarray, Yh: np.ndarray) -> list[float]:
    out = []
    for i in range(Y.shape[1]):
        y = Y[:, i]
        yh = Yh[:, i]
        sse = np.sum((y - yh) ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        out.append(float(1.0 - sse / max(sst, EPS)))
    return out


def _make_lag_dataset(z: np.ndarray, y: np.ndarray, seq: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    Y_list = []
    for sid in np.unique(seq):
        if sid < 0:
            continue
        idx = np.where(seq == sid)[0]
        if len(idx) <= lag:
            continue
        for j in range(lag, len(idx)):
            feats = [z[idx[j - k]] for k in range(0, lag + 1)]
            X_list.append(np.concatenate(feats, axis=0))
            Y_list.append(y[idx[j]])
    if len(X_list) == 0:
        return np.zeros((0, z.shape[1] * (lag + 1))), np.zeros((0, y.shape[1]))
    return np.vstack(X_list), np.vstack(Y_list)


def sufficiency_metrics(pack: Pack, seed: int = 42) -> dict[str, Any]:
    out = {}
    for lag in [0, 1, 2]:
        X, Y = _make_lag_dataset(pack.z, pack.vel, pack.seq, lag=lag)
        if len(X) < 100:
            out[f"lag{lag}_r2"] = float("nan")
            continue
        tr, te = _split_idx(len(X), seed=seed + lag)
        mu, sd = _zfit(X[tr])
        Xtr = _zapply(X[tr], mu, sd)
        Xte = _zapply(X[te], mu, sd)
        W = _ridge_fit(Xtr, Y[tr], l2=1e-3)
        Yh = _ridge_pred(Xte, W)
        out[f"lag{lag}_r2"] = _r2_global(Y[te], Yh)
    out["history_gain_lag2_minus_lag0"] = float(out["lag2_r2"] - out["lag0_r2"])
    return out


def _sequence_pairs(seq: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for sid in np.unique(seq):
        if sid < 0:
            continue
        idx = np.where(seq == sid)[0]
        if len(idx) < 2:
            continue
        pairs.append((idx[:-1], idx[1:]))
    return pairs


def linear_dynamics_metrics(pack: Pack, seed: int = 42) -> dict[str, Any]:
    pairs = _sequence_pairs(pack.seq)
    if len(pairs) == 0:
        return {"n_pairs": 0}
    i0 = np.concatenate([p[0] for p in pairs], axis=0)
    i1 = np.concatenate([p[1] for p in pairs], axis=0)

    X = pack.z[i0]
    Y = pack.z[i1]
    tr, te = _split_idx(len(X), seed=seed)
    mu, sd = _zfit(X[tr])
    Xtr = _zapply(X[tr], mu, sd)
    Xte = _zapply(X[te], mu, sd)
    W = _ridge_fit(Xtr, Y[tr], l2=1e-2)
    Yh = _ridge_pred(Xte, W)
    r2_lat = _r2_global(Y[te], Yh)

    # Also test next-velocity predictability from current latent as a proxy for internal motion model.
    Vnext = pack.vel[i1]
    Wv = _ridge_fit(Xtr, Vnext[tr], l2=1e-3)
    Vh = _ridge_pred(Xte, Wv)
    r2_vnext = _r2_global(Vnext[te], Vh)

    return {
        "n_pairs": int(len(X)),
        "latent_next_r2_global": float(r2_lat),
        "vel_next_from_latent_r2_global": float(r2_vnext),
    }


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float("nan")
    x = x - np.mean(x)
    den = float(np.dot(x, x))
    if den < EPS:
        return float("nan")
    return float(np.dot(x[:-lag], x[lag:]) / den)


def residual_whiteness_metrics(pack: Pack) -> dict[str, Any]:
    res = pack.pred - pack.vel
    ac_vals = {f"lag{lag}": [] for lag in range(1, 6)}
    for sid in np.unique(pack.seq):
        if sid < 0:
            continue
        idx = np.where(pack.seq == sid)[0]
        if len(idx) < 8:
            continue
        r = res[idx]
        for comp in range(3):
            x = r[:, comp]
            for lag in range(1, 6):
                a = _autocorr(x, lag)
                if np.isfinite(a):
                    ac_vals[f"lag{lag}"].append(a)

    out = {}
    abs_means = []
    for lag in range(1, 6):
        arr = np.asarray(ac_vals[f"lag{lag}"], dtype=np.float64)
        if len(arr) == 0:
            out[f"mean_ac_lag{lag}"] = float("nan")
            out[f"mean_abs_ac_lag{lag}"] = float("nan")
        else:
            out[f"mean_ac_lag{lag}"] = float(np.mean(arr))
            out[f"mean_abs_ac_lag{lag}"] = float(np.mean(np.abs(arr)))
            abs_means.append(np.mean(np.abs(arr)))
    out["whiteness_index_abs_ac_lag1to5"] = float(np.mean(abs_means)) if abs_means else float("nan")
    return out


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if den < EPS:
        return float("nan")
    return float(np.sum(rx * ry) / den)


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    # labels are {0,1}
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg], axis=0))
    rpos = np.sum(ranks[: len(pos)])
    auc = (rpos - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def _roc_curve_from_scores(scores: np.ndarray, labels: np.ndarray, n_thr: int = 120) -> dict[str, Any]:
    labels = labels.astype(int)
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return {"fpr": [], "tpr": [], "thresholds": []}
    smin, smax = float(np.min(scores)), float(np.max(scores))
    if abs(smax - smin) < EPS:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [smin, smax]}

    thr = np.linspace(smin, smax, n_thr)
    tpr = []
    fpr = []
    for th in thr:
        yhat = (scores >= th).astype(int)
        tp = np.sum((yhat == 1) & (labels == 1))
        fp = np.sum((yhat == 1) & (labels == 0))
        tpr.append(float(tp / pos))
        fpr.append(float(fp / neg))

    # include endpoints for nicer curves
    fpr = [0.0] + fpr + [1.0]
    tpr = [0.0] + tpr + [1.0]
    thresholds = [float("inf")] + [float(x) for x in thr] + [float("-inf")]
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def observability_proxy_metrics(pack: Pack) -> dict[str, Any]:
    e = np.linalg.norm(pack.pred - pack.vel, axis=1)
    z = pack.z

    # Proxy 1: latent norm
    p_norm = np.linalg.norm(z, axis=1)

    # Proxy 2: normalized Mahalanobis-like score (diag covariance).
    mu = z.mean(axis=0, keepdims=True)
    sd = z.std(axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    p_mah = np.mean(((z - mu) / sd) ** 2, axis=1)

    q_err = np.quantile(e, 0.75)
    high_err = (e >= q_err).astype(int)
    speed = np.linalg.norm(pack.vel, axis=1)
    q_speed_lo, q_speed_hi = np.quantile(speed, 0.25), np.quantile(speed, 0.75)
    low_speed = speed <= q_speed_lo
    high_speed = speed >= q_speed_hi

    q_ev_lo, q_ev_hi = np.quantile(pack.event_density, 0.25), np.quantile(pack.event_density, 0.75)
    sparse = pack.event_density <= q_ev_lo
    dense = pack.event_density >= q_ev_hi

    out = {
        "proxy_norm_spearman_err": _spearman(p_norm, e),
        "proxy_mah_spearman_err": _spearman(p_mah, e),
        "proxy_norm_auc_higherr": _auc_from_scores(p_norm, high_err),
        "proxy_mah_auc_higherr": _auc_from_scores(p_mah, high_err),
        "proxy_norm_sparse_over_dense": float(np.mean(p_norm[sparse]) / max(np.mean(p_norm[dense]), EPS)),
        "proxy_mah_sparse_over_dense": float(np.mean(p_mah[sparse]) / max(np.mean(p_mah[dense]), EPS)),
        "proxy_norm_highspeed_over_lowspeed": float(np.mean(p_norm[high_speed]) / max(np.mean(p_norm[low_speed]), EPS)),
        "proxy_mah_highspeed_over_lowspeed": float(np.mean(p_mah[high_speed]) / max(np.mean(p_mah[low_speed]), EPS)),
        "high_error_fraction": float(np.mean(high_err)),
        "roc_norm": _roc_curve_from_scores(p_norm, high_err),
        "roc_mah": _roc_curve_from_scores(p_mah, high_err),
    }
    return out


def _pca_reduce(X: np.ndarray, keep_var: float = 0.99) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(X) - 1, 1)
    csum = np.cumsum(var) / max(np.sum(var), EPS)
    k = int(np.searchsorted(csum, keep_var) + 1)
    k = max(1, min(k, U.shape[1]))
    return U[:, :k] * S[:k]


def cca_alignment_metrics(pack: Pack) -> dict[str, Any]:
    speed = np.linalg.norm(pack.vel, axis=1, keepdims=True)
    Y = np.concatenate([pack.vel, speed], axis=1)
    X = pack.z

    Xr = _pca_reduce(X, keep_var=0.99)
    Yr = _pca_reduce(Y, keep_var=1.0)
    Xr = Xr - Xr.mean(axis=0, keepdims=True)
    Yr = Yr - Yr.mean(axis=0, keepdims=True)
    n = len(Xr)

    Cxx = (Xr.T @ Xr) / max(n - 1, 1) + 1e-4 * np.eye(Xr.shape[1])
    Cyy = (Yr.T @ Yr) / max(n - 1, 1) + 1e-4 * np.eye(Yr.shape[1])
    Cxy = (Xr.T @ Yr) / max(n - 1, 1)

    wx, vx = np.linalg.eigh(Cxx)
    wy, vy = np.linalg.eigh(Cyy)
    wx = np.clip(wx, 1e-8, None)
    wy = np.clip(wy, 1e-8, None)
    Cxx_inv = vx @ np.diag(1.0 / np.sqrt(wx)) @ vx.T
    Cyy_inv = vy @ np.diag(1.0 / np.sqrt(wy)) @ vy.T
    M = Cxx_inv @ Cxy @ Cyy_inv
    cc = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)

    return {
        "cca_top1": float(cc[0]) if len(cc) > 0 else float("nan"),
        "cca_top2_mean": float(np.mean(cc[:2])) if len(cc) >= 2 else float("nan"),
        "cca_top3_mean": float(np.mean(cc[:3])) if len(cc) >= 3 else float("nan"),
        "cca_top4_mean": float(np.mean(cc[:4])) if len(cc) >= 4 else float("nan"),
        "cca_top8": [float(x) for x in cc[:8]],
    }


def pack_metrics(pack: Pack, seed: int = 42) -> dict[str, Any]:
    return {
        "name": pack.name,
        "n_samples": int(len(pack.z)),
        "state_sufficiency": sufficiency_metrics(pack, seed=seed),
        "linear_dynamics": linear_dynamics_metrics(pack, seed=seed),
        "residual_whiteness": residual_whiteness_metrics(pack),
        "observability_proxy": observability_proxy_metrics(pack),
        "physics_alignment": cca_alignment_metrics(pack),
    }


def _delta(a: float, b: float) -> float:
    return float(a - b)


def compare_metrics(flow: dict[str, Any], noflow: dict[str, Any]) -> dict[str, Any]:
    f_s = flow["state_sufficiency"]
    n_s = noflow["state_sufficiency"]
    f_d = flow["linear_dynamics"]
    n_d = noflow["linear_dynamics"]
    f_w = flow["residual_whiteness"]
    n_w = noflow["residual_whiteness"]
    f_o = flow["observability_proxy"]
    n_o = noflow["observability_proxy"]
    f_c = flow["physics_alignment"]
    n_c = noflow["physics_alignment"]

    return {
        "state_sufficiency": {
            "lag0_r2_delta": _delta(f_s["lag0_r2"], n_s["lag0_r2"]),
            "lag2_r2_delta": _delta(f_s["lag2_r2"], n_s["lag2_r2"]),
            "history_gain_delta": _delta(f_s["history_gain_lag2_minus_lag0"], n_s["history_gain_lag2_minus_lag0"]),
        },
        "linear_dynamics": {
            "latent_next_r2_delta": _delta(f_d["latent_next_r2_global"], n_d["latent_next_r2_global"]),
            "vel_next_from_latent_r2_delta": _delta(f_d["vel_next_from_latent_r2_global"], n_d["vel_next_from_latent_r2_global"]),
        },
        "residual_whiteness": {
            "whiteness_index_delta_flow_minus_noflow": _delta(
                f_w["whiteness_index_abs_ac_lag1to5"],
                n_w["whiteness_index_abs_ac_lag1to5"],
            )
        },
        "observability_proxy": {
            "proxy_norm_auc_higherr_delta": _delta(f_o["proxy_norm_auc_higherr"], n_o["proxy_norm_auc_higherr"]),
            "proxy_mah_auc_higherr_delta": _delta(f_o["proxy_mah_auc_higherr"], n_o["proxy_mah_auc_higherr"]),
            "proxy_norm_spearman_err_delta": _delta(f_o["proxy_norm_spearman_err"], n_o["proxy_norm_spearman_err"]),
            "proxy_mah_spearman_err_delta": _delta(f_o["proxy_mah_spearman_err"], n_o["proxy_mah_spearman_err"]),
        },
        "physics_alignment": {
            "cca_top1_delta": _delta(f_c["cca_top1"], n_c["cca_top1"]),
            "cca_top3_mean_delta": _delta(f_c["cca_top3_mean"], n_c["cca_top3_mean"]),
        },
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _score_dashboard(m: dict[str, Any]) -> dict[str, float]:
    # Normalize metrics into [0,1] for visual dashboard only.
    hg = m["state_sufficiency"]["history_gain_lag2_minus_lag0"]  # lower is better
    w = m["residual_whiteness"]["whiteness_index_abs_ac_lag1to5"]  # lower is better
    auc = m["observability_proxy"]["proxy_mah_auc_higherr"]  # higher is better
    cca = m["physics_alignment"]["cca_top1"]  # higher is better
    lat_dyn = m["linear_dynamics"]["latent_next_r2_global"]  # higher is better

    return {
        "State sufficiency": _clip01(1.0 - hg / 0.01),
        "Linear dynamics": _clip01((lat_dyn - 0.98) / 0.02),
        "Residual whiteness": _clip01(1.0 - w / 1.0),
        "Observability proxy": _clip01((auc - 0.5) / 0.5),
        "Physics alignment": _clip01((cca - 0.8) / 0.2),
    }


def _plot_single_dashboard(metrics: dict[str, Any], out_path: Path) -> None:
    if not HAS_MPL:
        return
    scores = _score_dashboard(metrics)
    labels = list(scores.keys())
    vals = np.array([scores[k] for k in labels], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    bars = ax.barh(y, vals, color="#0f766e")
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Normalized score [0,1]")
    ax.set_title("Classical-like Latent Signatures Dashboard")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7)
    for b, v in zip(bars, vals):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", ha="left")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_sufficiency(metrics: dict[str, Any], out_path: Path, compare: dict[str, Any] | None = None) -> None:
    if not HAS_MPL:
        return
    lags = np.array([0, 1, 2], dtype=int)
    y = np.array([
        metrics["state_sufficiency"]["lag0_r2"],
        metrics["state_sufficiency"]["lag1_r2"],
        metrics["state_sufficiency"]["lag2_r2"],
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(lags, y, marker="o", linewidth=2.2, color="#0f766e", label=metrics["name"])
    if compare is not None:
        y2 = np.array([
            compare["state_sufficiency"]["lag0_r2"],
            compare["state_sufficiency"]["lag1_r2"],
            compare["state_sufficiency"]["lag2_r2"],
        ], dtype=float)
        ax.plot(lags, y2, marker="o", linewidth=2.2, color="#b45309", label=compare["name"])

    ax.set_xticks(lags, [f"lag {k}" for k in lags])
    ax.set_ylabel("Velocity probe $R^2$")
    ax.set_title("State Sufficiency vs Added History")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_residual_ac(metrics: dict[str, Any], out_path: Path, compare: dict[str, Any] | None = None) -> None:
    if not HAS_MPL:
        return
    lags = np.arange(1, 6)
    y = np.array([metrics["residual_whiteness"][f"mean_abs_ac_lag{k}"] for k in lags], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(lags, y, marker="o", linewidth=2.2, color="#0f766e", label=metrics["name"])
    if compare is not None:
        y2 = np.array([compare["residual_whiteness"][f"mean_abs_ac_lag{k}"] for k in lags], dtype=float)
        ax.plot(lags, y2, marker="o", linewidth=2.2, color="#b45309", label=compare["name"])
    ax.set_xticks(lags)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Mean |autocorrelation|")
    ax.set_title("Residual Whiteness Profile")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_observability_roc(metrics: dict[str, Any], out_path: Path, compare: dict[str, Any] | None = None) -> None:
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    rn = metrics["observability_proxy"]["roc_norm"]
    rm = metrics["observability_proxy"]["roc_mah"]
    ax.plot(rn["fpr"], rn["tpr"], color="#0f766e", linewidth=2.2, label=f"{metrics['name']} norm (AUC={metrics['observability_proxy']['proxy_norm_auc_higherr']:.3f})")
    ax.plot(rm["fpr"], rm["tpr"], color="#0ea5e9", linewidth=2.2, label=f"{metrics['name']} mah (AUC={metrics['observability_proxy']['proxy_mah_auc_higherr']:.3f})")
    if compare is not None:
        rn2 = compare["observability_proxy"]["roc_norm"]
        rm2 = compare["observability_proxy"]["roc_mah"]
        ax.plot(rn2["fpr"], rn2["tpr"], color="#b45309", linewidth=2.2, linestyle="--", label=f"{compare['name']} norm (AUC={compare['observability_proxy']['proxy_norm_auc_higherr']:.3f})")
        ax.plot(rm2["fpr"], rm2["tpr"], color="#ef4444", linewidth=2.2, linestyle="--", label=f"{compare['name']} mah (AUC={compare['observability_proxy']['proxy_mah_auc_higherr']:.3f})")

    ax.plot([0, 1], [0, 1], color="#64748b", linestyle=":", linewidth=1.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("High-Error Detection ROC from Latent Proxies")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_cca(metrics: dict[str, Any], out_path: Path, compare: dict[str, Any] | None = None) -> None:
    if not HAS_MPL:
        return
    cc = np.asarray(metrics["physics_alignment"]["cca_top8"], dtype=float)
    x = np.arange(1, len(cc) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(x, cc, marker="o", linewidth=2.2, color="#0f766e", label=metrics["name"])
    if compare is not None:
        cc2 = np.asarray(compare["physics_alignment"]["cca_top8"], dtype=float)
        x2 = np.arange(1, len(cc2) + 1)
        ax.plot(x2, cc2, marker="o", linewidth=2.2, color="#b45309", label=compare["name"])

    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Canonical correlation")
    ax.set_title("Latent-to-Physics CCA Spectrum")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_regime_ratios(metrics: dict[str, Any], out_path: Path, compare: dict[str, Any] | None = None) -> None:
    if not HAS_MPL:
        return
    keys = [
        "proxy_norm_sparse_over_dense",
        "proxy_mah_sparse_over_dense",
        "proxy_norm_highspeed_over_lowspeed",
        "proxy_mah_highspeed_over_lowspeed",
    ]
    labels = ["norm sparse/dense", "mah sparse/dense", "norm high/low speed", "mah high/low speed"]
    v1 = np.array([metrics["observability_proxy"][k] for k in keys], dtype=float)
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    if compare is None:
        ax.bar(x, v1, color="#0f766e")
    else:
        v2 = np.array([compare["observability_proxy"][k] for k in keys], dtype=float)
        ax.bar(x - w / 2, v1, width=w, color="#0f766e", label=metrics["name"])
        ax.bar(x + w / 2, v2, width=w, color="#b45309", label=compare["name"])
        ax.legend()
    ax.axhline(1.0, color="#64748b", linestyle=":", linewidth=1.2)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_title("Latent Proxy Inflation in Hard Regimes")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_plots(out_dir: Path, flow: dict[str, Any], noflow: dict[str, Any] | None = None) -> None:
    if not HAS_MPL:
        return
    _setup_plot_style()
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    _plot_single_dashboard(flow, pdir / "dashboard_single.png")
    _plot_sufficiency(flow, pdir / "sufficiency_lag_curve.png", compare=noflow)
    _plot_residual_ac(flow, pdir / "residual_autocorr_profile.png", compare=noflow)
    _plot_observability_roc(flow, pdir / "observability_roc.png", compare=noflow)
    _plot_cca(flow, pdir / "cca_spectrum.png", compare=noflow)
    _plot_regime_ratios(flow, pdir / "regime_proxy_ratios.png", compare=noflow)


def _single_verdicts(m: dict[str, Any]) -> dict[str, str]:
    out = {}
    hg = m["state_sufficiency"]["history_gain_lag2_minus_lag0"]
    if hg < 0.002:
        out["state_sufficiency"] = "strong"
    elif hg < 0.004:
        out["state_sufficiency"] = "moderate"
    else:
        out["state_sufficiency"] = "weak"

    lz = m["linear_dynamics"]["latent_next_r2_global"]
    if lz > 0.995:
        out["linear_dynamics"] = "strong"
    elif lz > 0.99:
        out["linear_dynamics"] = "moderate"
    else:
        out["linear_dynamics"] = "weak"

    w = m["residual_whiteness"]["whiteness_index_abs_ac_lag1to5"]
    if w < 0.65:
        out["residual_whiteness"] = "strong"
    elif w < 0.8:
        out["residual_whiteness"] = "moderate"
    else:
        out["residual_whiteness"] = "weak"

    auc = m["observability_proxy"]["proxy_mah_auc_higherr"]
    if auc > 0.8:
        out["observability_proxy"] = "strong"
    elif auc > 0.75:
        out["observability_proxy"] = "moderate"
    else:
        out["observability_proxy"] = "weak"

    cca = m["physics_alignment"]["cca_top1"]
    if cca > 0.985:
        out["physics_alignment"] = "strong"
    elif cca > 0.96:
        out["physics_alignment"] = "moderate"
    else:
        out["physics_alignment"] = "weak"
    return out


def save_report(path: Path, flow: dict[str, Any], noflow: dict[str, Any] | None, delta: dict[str, Any] | None) -> None:
    def fmt(x: float) -> str:
        if x is None or not np.isfinite(x):
            return "N/A"
        return f"{x:.6f}"

    lines = []
    lines.append("# Classical-Hint Analysis from Latents")
    lines.append("")
    lines.append(f"- Primary model: `{flow['name']}`")
    if noflow is not None:
        lines.append(f"- Comparator model: `{noflow['name']}`")
    else:
        lines.append("- Comparator model: `None (single-model mode)`")
    lines.append("")
    lines.append("## 1) State Sufficiency")
    lines.append("")
    if noflow is None:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k in ["lag0_r2", "lag1_r2", "lag2_r2", "history_gain_lag2_minus_lag0"]:
            lines.append(f"| {k} | {fmt(flow['state_sufficiency'][k])} |")
    else:
        lines.append("| Metric | Flow | No-flow | Delta |")
        lines.append("|---|---:|---:|---:|")
        for k in ["lag0_r2", "lag1_r2", "lag2_r2", "history_gain_lag2_minus_lag0"]:
            lines.append(
                f"| {k} | {fmt(flow['state_sufficiency'][k])} | {fmt(noflow['state_sufficiency'][k])} | {fmt(flow['state_sufficiency'][k]-noflow['state_sufficiency'][k])} |"
            )

    lines.append("")
    lines.append("## 2) Local Linear Dynamics")
    lines.append("")
    if noflow is None:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k in ["latent_next_r2_global", "vel_next_from_latent_r2_global"]:
            lines.append(f"| {k} | {fmt(flow['linear_dynamics'][k])} |")
    else:
        lines.append("| Metric | Flow | No-flow | Delta |")
        lines.append("|---|---:|---:|---:|")
        for k in ["latent_next_r2_global", "vel_next_from_latent_r2_global"]:
            lines.append(
                f"| {k} | {fmt(flow['linear_dynamics'][k])} | {fmt(noflow['linear_dynamics'][k])} | {fmt(flow['linear_dynamics'][k]-noflow['linear_dynamics'][k])} |"
            )

    lines.append("")
    lines.append("## 3) Innovation-Like Residuals")
    lines.append("")
    if noflow is None:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
    else:
        lines.append("| Metric | Flow | No-flow | Delta |")
        lines.append("|---|---:|---:|---:|")
    k = "whiteness_index_abs_ac_lag1to5"
    if noflow is None:
        lines.append(f"| {k} | {fmt(flow['residual_whiteness'][k])} |")
    else:
        lines.append(
            f"| {k} | {fmt(flow['residual_whiteness'][k])} | {fmt(noflow['residual_whiteness'][k])} | {fmt(flow['residual_whiteness'][k]-noflow['residual_whiteness'][k])} |"
        )
    lines.append("")
    lines.append("Smaller whiteness index is better (less residual autocorrelation).")

    lines.append("")
    lines.append("## 4) Observability-Aware Proxy")
    lines.append("")
    keys_obs = [
        "proxy_norm_spearman_err",
        "proxy_mah_spearman_err",
        "proxy_norm_auc_higherr",
        "proxy_mah_auc_higherr",
        "proxy_norm_sparse_over_dense",
        "proxy_mah_sparse_over_dense",
        "proxy_norm_highspeed_over_lowspeed",
        "proxy_mah_highspeed_over_lowspeed",
    ]
    if noflow is None:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k in keys_obs:
            lines.append(f"| {k} | {fmt(flow['observability_proxy'][k])} |")
    else:
        lines.append("| Metric | Flow | No-flow | Delta |")
        lines.append("|---|---:|---:|---:|")
        for k in keys_obs:
            lines.append(
                f"| {k} | {fmt(flow['observability_proxy'][k])} | {fmt(noflow['observability_proxy'][k])} | {fmt(flow['observability_proxy'][k]-noflow['observability_proxy'][k])} |"
            )

    lines.append("")
    lines.append("## 5) Alignment with Physical Variables")
    lines.append("")
    keys_cca = ["cca_top1", "cca_top2_mean", "cca_top3_mean", "cca_top4_mean"]
    if noflow is None:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k in keys_cca:
            lines.append(f"| {k} | {fmt(flow['physics_alignment'][k])} |")
    else:
        lines.append("| Metric | Flow | No-flow | Delta |")
        lines.append("|---|---:|---:|---:|")
        for k in keys_cca:
            lines.append(
                f"| {k} | {fmt(flow['physics_alignment'][k])} | {fmt(noflow['physics_alignment'][k])} | {fmt(flow['physics_alignment'][k]-noflow['physics_alignment'][k])} |"
            )

    lines.append("")
    lines.append("## Claim-Oriented Readout")
    lines.append("")
    if noflow is None:
        verdicts = _single_verdicts(flow)
        lines.append(f"- State sufficiency: `{verdicts['state_sufficiency']}`")
        lines.append(f"- Local linear dynamics: `{verdicts['linear_dynamics']}`")
        lines.append(f"- Innovation-like residuals: `{verdicts['residual_whiteness']}`")
        lines.append(f"- Observability-aware proxy: `{verdicts['observability_proxy']}`")
        lines.append(f"- Physical alignment: `{verdicts['physics_alignment']}`")
    else:
        lines.append(
            f"- If `history_gain_lag2_minus_lag0` is small, latent is closer to a sufficient state summary."
        )
        lines.append(
            f"- Higher `latent_next_r2_global` and `vel_next_from_latent_r2_global` indicate more linear state transition structure."
        )
        lines.append(
            f"- Lower `whiteness_index_abs_ac_lag1to5` indicates more innovation-like residuals."
        )
        lines.append(
            f"- Higher AUC/Spearman for observability proxies indicates latent confidence proxy tracks error/hard regimes."
        )
        lines.append(
            f"- Higher CCA indicates stronger latent alignment with physical kinematic variables."
        )

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `plots/dashboard_single.png`")
    lines.append("- `plots/sufficiency_lag_curve.png`")
    lines.append("- `plots/residual_autocorr_profile.png`")
    lines.append("- `plots/observability_roc.png`")
    lines.append("- `plots/cca_spectrum.png`")
    lines.append("- `plots/regime_proxy_ratios.png`")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect latent hints of classical-like behavior.")
    parser.add_argument("--flow", required=True, help="Path to flow extracted .npz")
    parser.add_argument("--noflow", default=None, help="Optional path to comparison extracted .npz")
    parser.add_argument("--out", default="plots/classical_hints", help="Output directory")
    parser.add_argument("--flow-name", default="with_flow")
    parser.add_argument("--noflow-name", default="without_flow")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_MPL:
        print("Warning: matplotlib not available, plots will not be generated.")

    flow_pack = load_pack(Path(args.flow), args.flow_name)
    flow_metrics = pack_metrics(flow_pack, seed=args.seed)
    save_json(out_dir / f"metrics_{args.flow_name}.json", flow_metrics)
    if args.noflow is not None:
        noflow_pack = load_pack(Path(args.noflow), args.noflow_name)
        noflow_metrics = pack_metrics(noflow_pack, seed=args.seed + 101)
        deltas = compare_metrics(flow_metrics, noflow_metrics)
        save_json(out_dir / f"metrics_{args.noflow_name}.json", noflow_metrics)
        save_json(out_dir / "metrics_delta.json", deltas)
        save_report(out_dir / "report.md", flow_metrics, noflow_metrics, deltas)
        generate_plots(out_dir, flow_metrics, noflow=noflow_metrics)
        print(f"Saved outputs to: {out_dir}")
        print("Key deltas:")
        print(json.dumps(deltas, indent=2))
    else:
        save_report(out_dir / "report.md", flow_metrics, None, None)
        verdicts = _single_verdicts(flow_metrics)
        save_json(out_dir / "verdicts_single_model.json", verdicts)
        generate_plots(out_dir, flow_metrics, noflow=None)
        print(f"Saved outputs to: {out_dir}")
        print("Single-model verdicts:")
        print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
