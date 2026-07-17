import argparse
import json
import warnings

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

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    TSNE = None

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    umap = None

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    px = None


EPS = 1e-12


@dataclass
class LatentData:
    name: str
    z: np.ndarray
    pred: np.ndarray
    vel: np.ndarray
    pos: np.ndarray
    times: np.ndarray
    seq: np.ndarray
    event_density: np.ndarray


def _setup_style() -> None:
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
            "grid.alpha": 0.6,
            "font.size": 11,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#cbd5e1",
        }
    )


def _parse_seq_ids(values: np.ndarray) -> np.ndarray:
    vals = values.reshape(-1)
    if vals.dtype.kind in {"U", "S", "O"}:
        out = []
        for x in vals:
            sx = str(x)
            out.append(int(sx) if sx.isdigit() else -1)
        return np.asarray(out, dtype=np.int32)
    return vals.astype(np.int32)


def _require_key(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz.files:
        raise KeyError(f"Missing key `{key}` in {npz.files}")
    return npz[key]


def load_extracted(path: Path, name: str) -> LatentData:
    npz = np.load(path, allow_pickle=True)
    z = np.asarray(_require_key(npz, "fused"), dtype=np.float64)
    pred = np.asarray(_require_key(npz, "pred"), dtype=np.float64)
    vel = np.asarray(_require_key(npz, "target_vel"), dtype=np.float64)
    pos = np.asarray(_require_key(npz, "target_pos"), dtype=np.float64)
    times = np.asarray(_require_key(npz, "times"), dtype=np.float64).reshape(-1)
    seq = _parse_seq_ids(_require_key(npz, "sequence_id"))
    event_density = np.asarray(_require_key(npz, "event_density"), dtype=np.float64).reshape(-1)

    n = len(z)
    for arr_name, arr in [
        ("pred", pred),
        ("vel", vel),
        ("pos", pos),
        ("times", times),
        ("seq", seq),
        ("event_density", event_density),
    ]:
        if len(arr) != n:
            raise ValueError(f"Inconsistent array lengths: fused={n}, {arr_name}={len(arr)}")

    return LatentData(
        name=name,
        z=z,
        pred=pred,
        vel=vel,
        pos=pos,
        times=times,
        seq=seq,
        event_density=event_density,
    )


def _parse_seq_arg(s: str | None) -> set[int] | None:
    if s is None or s.strip() == "":
        return None
    out = set()
    for token in s.split(","):
        token = token.strip()
        if token == "":
            continue
        out.add(int(token))
    return out


def filter_data(data: LatentData, seq_filter: set[int] | None) -> LatentData:
    if seq_filter is None:
        return data
    mask = np.array([int(s) in seq_filter for s in data.seq], dtype=bool)
    if mask.sum() == 0:
        raise ValueError("Sequence filter produced zero samples.")
    return LatentData(
        name=data.name,
        z=data.z[mask],
        pred=data.pred[mask],
        vel=data.vel[mask],
        pos=data.pos[mask],
        times=data.times[mask],
        seq=data.seq[mask],
        event_density=data.event_density[mask],
    )


def subsample_data(data: LatentData, max_samples: int | None, seed: int) -> LatentData:
    if max_samples is None or len(data.z) <= max_samples:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data.z), size=max_samples, replace=False)
    idx.sort()
    return LatentData(
        name=data.name,
        z=data.z[idx],
        pred=data.pred[idx],
        vel=data.vel[idx],
        pos=data.pos[idx],
        times=data.times[idx],
        seq=data.seq[idx],
        event_density=data.event_density[idx],
    )


def _pca_project(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :n_components] * S[:n_components]
    var = (S ** 2) / max(len(X) - 1, 1)
    var_ratio = var / max(np.sum(var), EPS)
    return Z, var_ratio


def _reduce_for_nonlinear(X: np.ndarray, k: int = 50) -> np.ndarray:
    k = min(k, X.shape[1], X.shape[0] - 1)
    if k <= 0:
        return X
    Z, _ = _pca_project(X, n_components=k)
    return Z


def project_manifold(
    X: np.ndarray,
    method: str,
    dim: int,
    seed: int,
    tsne_perplexity: float,
    umap_neighbors: int,
    umap_min_dist: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    method = method.lower()
    meta: dict[str, Any] = {"requested_method": method, "dim": dim}

    if method == "auto":
        if HAS_UMAP:
            method = "umap"
        elif HAS_SKLEARN:
            method = "tsne"
        else:
            method = "pca"
        meta["resolved_method"] = method
    else:
        meta["resolved_method"] = method

    if method == "pca":
        Z, var_ratio = _pca_project(X, n_components=dim)
        meta["explained_variance"] = [float(x) for x in var_ratio[:dim]]
        return Z, meta

    if method == "tsne":
        if not HAS_SKLEARN:
            warnings.warn("scikit-learn not available. Falling back to PCA.")
            Z, var_ratio = _pca_project(X, n_components=dim)
            meta["resolved_method"] = "pca_fallback"
            meta["explained_variance"] = [float(x) for x in var_ratio[:dim]]
            return Z, meta
        Xr = _reduce_for_nonlinear(X, k=50)
        perpl = min(tsne_perplexity, max(5.0, (len(Xr) - 1) / 3.0))
        tsne = TSNE(
            n_components=dim,
            perplexity=perpl,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            n_iter=1200,
            verbose=0,
        )
        Z = tsne.fit_transform(Xr)
        meta["tsne_perplexity"] = float(perpl)
        return Z, meta

    if method == "umap":
        if not HAS_UMAP:
            warnings.warn("umap-learn not available. Falling back to PCA.")
            Z, var_ratio = _pca_project(X, n_components=dim)
            meta["resolved_method"] = "pca_fallback"
            meta["explained_variance"] = [float(x) for x in var_ratio[:dim]]
            return Z, meta
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            random_state=seed,
            metric="euclidean",
        )
        Z = reducer.fit_transform(X)
        meta["umap_neighbors"] = int(umap_neighbors)
        meta["umap_min_dist"] = float(umap_min_dist)
        return Z, meta

    raise ValueError(f"Unsupported method: {method}")


def _continuous_scatter_2d(
    Z: np.ndarray,
    c: np.ndarray,
    title: str,
    cbar: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    im = ax.scatter(Z[:, 0], Z[:, 1], c=c, s=8, cmap=cmap, alpha=0.78, linewidths=0)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(cbar)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _continuous_scatter_3d(
    Z3: np.ndarray,
    c: np.ndarray,
    title: str,
    cbar: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    if not HAS_MPL:
        return
    fig = plt.figure(figsize=(8.2, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    im = ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=c, s=7, cmap=cmap, alpha=0.72, linewidths=0)
    ax.set_xlabel("Comp 1")
    ax.set_ylabel("Comp 2")
    ax.set_zlabel("Comp 3")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(cbar)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _hex_landscape(
    Z: np.ndarray,
    value: np.ndarray,
    title: str,
    cbar: str,
    out_path: Path,
    cmap: str = "magma",
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    hb = ax.hexbin(
        Z[:, 0],
        Z[:, 1],
        C=value,
        reduce_C_function=np.mean,
        gridsize=48,
        mincnt=8,
        cmap=cmap,
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7)
    cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(cbar)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _pick_sequences_for_traj(seq: np.ndarray, err: np.ndarray, num_traj: int) -> list[int]:
    ids = [int(s) for s in np.unique(seq) if int(s) >= 0]
    if len(ids) <= num_traj:
        return sorted(ids)
    scores = []
    for sid in ids:
        mask = seq == sid
        if mask.sum() < 20:
            continue
        scores.append((sid, float(np.mean(err[mask]))))
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = [sid for sid, _ in scores[:num_traj]]
    return sorted(chosen)


def _traj_plot_2d(Z: np.ndarray, seq: np.ndarray, times: np.ndarray, err: np.ndarray, out_path: Path, num_traj: int) -> list[int]:
    if not HAS_MPL:
        return []
    chosen = _pick_sequences_for_traj(seq, err, num_traj=num_traj)
    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, sid in enumerate(chosen):
        idx = np.where(seq == sid)[0]
        order = np.argsort(times[idx])
        idx = idx[order]
        z = Z[idx]
        color = cmap(i % 10)
        ax.plot(z[:, 0], z[:, 1], color=color, linewidth=2.0, alpha=0.9, label=f"{sid:04d}")
        ax.scatter(z[0, 0], z[0, 1], color=color, s=35, marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter(z[-1, 0], z[-1, 1], color=color, s=45, marker="X", edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Latent Trajectories (start=o, end=X)")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(title="Sequence", ncol=2, fontsize=8)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return chosen


def _traj_plot_3d(Z3: np.ndarray, seq: np.ndarray, times: np.ndarray, err: np.ndarray, out_path: Path, chosen: list[int] | None = None, num_traj: int = 6) -> list[int]:
    if not HAS_MPL:
        return []
    if chosen is None:
        chosen = _pick_sequences_for_traj(seq, err, num_traj=num_traj)
    fig = plt.figure(figsize=(9.0, 7.3), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    for i, sid in enumerate(chosen):
        idx = np.where(seq == sid)[0]
        order = np.argsort(times[idx])
        idx = idx[order]
        z = Z3[idx]
        color = cmap(i % 10)
        ax.plot(z[:, 0], z[:, 1], z[:, 2], color=color, linewidth=2.0, alpha=0.95, label=f"{sid:04d}")
        ax.scatter(z[0, 0], z[0, 1], z[0, 2], color=color, s=30, marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter(z[-1, 0], z[-1, 1], z[-1, 2], color=color, s=42, marker="X", edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Comp 1")
    ax.set_ylabel("Comp 2")
    ax.set_zlabel("Comp 3")
    ax.set_title("3D Latent Trajectories (hard sequences)")
    ax.legend(title="Sequence", fontsize=8, ncol=2, loc="upper right")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return chosen


def _interactive_plotly_3d(
    Z3: np.ndarray,
    data: LatentData,
    speed: np.ndarray,
    err: np.ndarray,
    out_dir: Path,
) -> None:
    if not HAS_PLOTLY:
        return
    df = {
        "c1": Z3[:, 0],
        "c2": Z3[:, 1],
        "c3": Z3[:, 2],
        "speed": speed,
        "error": err,
        "seq": [f"{int(s):04d}" if int(s) >= 0 else "unk" for s in data.seq],
    }
    fig1 = px.scatter_3d(
        df,
        x="c1",
        y="c2",
        z="c3",
        color="speed",
        color_continuous_scale="Viridis",
        opacity=0.72,
        title=f"{data.name}: 3D manifold colored by speed",
    )
    fig1.write_html(str(out_dir / "manifold_3d_speed.html"), include_plotlyjs="cdn")

    fig2 = px.scatter_3d(
        df,
        x="c1",
        y="c2",
        z="c3",
        color="error",
        color_continuous_scale="Turbo",
        opacity=0.72,
        title=f"{data.name}: 3D manifold colored by error",
    )
    fig2.write_html(str(out_dir / "manifold_3d_error.html"), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aesthetic 2D/3D latent manifold visualizations.")
    parser.add_argument("--npz", required=True, help="Path to extracted latent .npz (e.g. extracted_with_flow_best.npz)")
    parser.add_argument("--name", default="with_flow_best")
    parser.add_argument("--out", default="plots/latent_manifold")
    parser.add_argument("--method", default="auto", choices=["auto", "pca", "tsne", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=7000)
    parser.add_argument("--sequences", default=None, help="Optional comma-separated sequence ids, e.g. 0010,0019,0024")
    parser.add_argument("--num-traj", type=int, default=8, help="How many hard trajectories to draw")
    parser.add_argument("--tsne-perplexity", type=float, default=35.0)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.08)
    parser.add_argument("--interactive", action="store_true", help="Export interactive Plotly 3D html")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if HAS_MPL:
        _setup_style()
    else:
        warnings.warn("matplotlib not available; static plots will be skipped.")

    data = load_extracted(Path(args.npz), name=args.name)
    seq_filter = _parse_seq_arg(args.sequences)
    data = filter_data(data, seq_filter)
    data = subsample_data(data, max_samples=args.max_samples, seed=args.seed)

    speed = np.linalg.norm(data.vel, axis=1)
    err = np.linalg.norm(data.pred - data.vel, axis=1)
    latent_norm = np.linalg.norm(data.z, axis=1)

    # 2D manifold for landscapes
    Z2, meta2 = project_manifold(
        data.z,
        method=args.method,
        dim=2,
        seed=args.seed,
        tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    # 3D manifold (same method request; for trajectories we also keep a PCA 3D fallback)
    Z3, meta3 = project_manifold(
        data.z,
        method=args.method,
        dim=3,
        seed=args.seed,
        tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    Z3_pca, pca_var = _pca_project(data.z, n_components=3)

    # Static plots
    _continuous_scatter_2d(
        Z2,
        speed,
        f"{data.name}: 2D manifold (speed)",
        "speed",
        out_dir / "manifold_2d_speed.png",
    )
    _continuous_scatter_2d(
        Z2,
        err,
        f"{data.name}: 2D manifold (prediction error)",
        "error",
        out_dir / "manifold_2d_error.png",
        cmap="turbo",
    )
    _continuous_scatter_2d(
        Z2,
        data.event_density,
        f"{data.name}: 2D manifold (event density)",
        "event density",
        out_dir / "manifold_2d_event_density.png",
        cmap="cividis",
    )
    _hex_landscape(
        Z2,
        err,
        f"{data.name}: error landscape over manifold",
        "mean error",
        out_dir / "manifold_error_landscape_hexbin.png",
    )
    _hex_landscape(
        Z2,
        speed,
        f"{data.name}: speed landscape over manifold",
        "mean speed",
        out_dir / "manifold_speed_landscape_hexbin.png",
        cmap="viridis",
    )
    _hex_landscape(
        Z2,
        data.event_density,
        f"{data.name}: event-density landscape over manifold",
        "mean event density",
        out_dir / "manifold_eventdensity_landscape_hexbin.png",
        cmap="cividis",
    )

    _continuous_scatter_3d(
        Z3,
        speed,
        f"{data.name}: 3D manifold (speed)",
        "speed",
        out_dir / "manifold_3d_speed.png",
    )
    _continuous_scatter_3d(
        Z3,
        err,
        f"{data.name}: 3D manifold (error)",
        "error",
        out_dir / "manifold_3d_error.png",
        cmap="turbo",
    )

    chosen = _traj_plot_2d(Z2, data.seq, data.times, err, out_dir / "latent_trajectories_2d.png", num_traj=args.num_traj)
    _traj_plot_3d(Z3_pca, data.seq, data.times, err, out_dir / "latent_trajectories_3d_pca.png", chosen=chosen, num_traj=args.num_traj)

    if args.interactive:
        _interactive_plotly_3d(Z3, data, speed, err, out_dir)

    summary = {
        "name": data.name,
        "n_samples": int(len(data.z)),
        "seq_filter": sorted(seq_filter) if seq_filter is not None else None,
        "method_2d": meta2,
        "method_3d": meta3,
        "pca3d_explained_variance": [float(x) for x in pca_var[:3]],
        "chosen_trajectory_sequences": [int(x) for x in chosen],
        "error_stats": {
            "mean": float(np.mean(err)),
            "median": float(np.median(err)),
            "q75": float(np.quantile(err, 0.75)),
            "q90": float(np.quantile(err, 0.90)),
        },
        "speed_stats": {
            "mean": float(np.mean(speed)),
            "median": float(np.median(speed)),
            "q75": float(np.quantile(speed, 0.75)),
            "q90": float(np.quantile(speed, 0.90)),
        },
        "latent_norm_stats": {
            "mean": float(np.mean(latent_norm)),
            "median": float(np.median(latent_norm)),
        },
        "has_matplotlib": bool(HAS_MPL),
        "has_sklearn": bool(HAS_SKLEARN),
        "has_umap": bool(HAS_UMAP),
        "has_plotly": bool(HAS_PLOTLY),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved visualizations to: {out_dir}")
    print(f"Used method 2D: {meta2.get('resolved_method', meta2.get('requested_method'))}")
    print(f"Used method 3D: {meta3.get('resolved_method', meta3.get('requested_method'))}")
    print(f"PCA 3D variance: {[round(float(v), 4) for v in pca_var[:3]]}")


if __name__ == "__main__":
    main()
