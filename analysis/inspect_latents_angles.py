import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


def _load_latent_files(path: Path) -> list[dict]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found at {path}")

    data = []
    for f in files:
        npz = np.load(f, allow_pickle=True)
        sample = {
            "fused": npz["fused"],
            "pred": npz["pred"],
            "target_vel": npz["target_vel"],
            "target_pos": npz["target_pos"],
            "times": npz["times"],
            "event_tokens": npz.get("event_tokens", None),
            "total_tokens": npz.get("total_tokens", None),
            "attention": npz.get("attention", None),
        }
        # Normalize optional fields that may be stored as object None
        if sample["attention"] is not None:
            if sample["attention"].dtype == object and sample["attention"].shape == ():
                sample["attention"] = None
        data.append(sample)
    return data


def _concat(data: list[dict], key: str):
    arrays = [d[key] for d in data if d.get(key) is not None]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def _linear_probe_r2(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    X_train = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    W, *_ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    Y_pred = X_test @ W

    ss_res = np.sum((Y_test - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "r2_vx": float(r2[0]),
        "r2_vy": float(r2[1]),
        "r2_vz": float(r2[2]),
        "r2_mean": float(r2.mean()),
    }


def _pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]


def _plot_pca(X_2d: np.ndarray, color: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color, s=6, cmap="viridis", alpha=0.7)
    plt.colorbar(label=title)
    plt.title(f"Latent PCA colored by {title}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_corr_heatmap(corr: np.ndarray, labels: list[str], out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.yticks(range(corr.shape[0]), labels)
    plt.xlabel("Latent dim")
    plt.title("Correlation: targets vs latent dims")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _attention_summary(attention: np.ndarray, event_tokens: int) -> dict:
    # Attention expected shapes:
    # (B, tgt, src) or (tgt, src) or (B, heads, tgt, src)
    att = attention
    if att.ndim == 4:
        att_mean = att.mean(axis=(0, 1))
    elif att.ndim == 3:
        att_mean = att.mean(axis=0)
    else:
        att_mean = att

    # Aggregate by modality groups (event tokens + imu + range + angles)
    e_end = event_tokens
    imu_idx = e_end
    range_idx = e_end + 1
    angle_idx = e_end + 2

    modality_attn = {
        "event": float(att_mean[:, :e_end].mean()) if e_end > 0 else float("nan"),
        "imu": float(att_mean[:, imu_idx].mean()) if imu_idx < att_mean.shape[1] else float("nan"),
        "range": float(att_mean[:, range_idx].mean()) if range_idx < att_mean.shape[1] else float("nan"),
        "angle": float(att_mean[:, angle_idx].mean()) if angle_idx < att_mean.shape[1] else float("nan"),
    }

    return {"attn_mean": att_mean, "modality_attn": modality_attn}


def main():
    parser = argparse.ArgumentParser(description="Inspect fused latents and attention logs.")
    parser.add_argument("--path", required=True, help="Path to a latents .npz or directory")
    parser.add_argument("--out", default=None, help="Output directory for plots")
    parser.add_argument("--max_dims", type=int, default=64, help="Max latent dims for correlation plot")
    args = parser.parse_args()

    path = Path(args.path)
    out_dir = Path(args.out) if args.out else (path if path.is_dir() else path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_latent_files(path)
    fused = _concat(data, "fused")
    pred = _concat(data, "pred")
    target_vel = _concat(data, "target_vel")
    target_pos = _concat(data, "target_pos")
    times = _concat(data, "times")
    attention = _concat(data, "attention")

    if fused is None or target_vel is None:
        raise RuntimeError("Missing fused or target_vel in latent files.")

    speed = np.linalg.norm(target_vel, axis=1)

    # Linear probe
    r2 = _linear_probe_r2(fused, target_vel)
    print("Linear probe R2 (velocity):", r2)

    # PCA scatter
    X_2d = _pca_2d(fused)
    _plot_pca(X_2d, speed, out_dir / "latent_pca_speed.png", "speed")
    _plot_pca(X_2d, target_vel[:, 2], out_dir / "latent_pca_vz.png", "vz")

    # Correlation heatmap (targets vs latent dims)
    max_dims = min(args.max_dims, fused.shape[1])
    corr_targets = np.stack(
        [
            target_vel[:, 0],
            target_vel[:, 1],
            target_vel[:, 2],
            speed,
        ],
        axis=0,
    )
    # Build a (N, 4 + D) matrix and compute column-wise correlation
    corr_mat = np.corrcoef(
        np.concatenate([corr_targets.T, fused[:, :max_dims]], axis=1),
        rowvar=False
    )
    corr = corr_mat[:4, 4:]
    _plot_corr_heatmap(
        corr,
        ["vx", "vy", "vz", "speed"],
        out_dir / "latent_corr.png"
    )

    # Top correlated dims (speed and vz)
    corr_speed = corr[3]
    corr_vz = corr[2]
    top_speed = np.argsort(np.abs(corr_speed))[-10:][::-1]
    top_vz = np.argsort(np.abs(corr_vz))[-10:][::-1]
    print("Top latent dims by |corr| with speed:", top_speed.tolist())
    print("Top latent dims by |corr| with vz:", top_vz.tolist())

    # Attention analysis
    if attention is not None:
        event_tokens = _concat(data, "event_tokens")
        if event_tokens is not None and np.any(event_tokens > 0):
            event_tokens_val = int(np.median(event_tokens[event_tokens > 0]))
        else:
            event_tokens_val = 4

        attn_summary = _attention_summary(attention, event_tokens_val)
        attn_mean = attn_summary["attn_mean"]
        modality_attn = attn_summary["modality_attn"]

        plt.figure(figsize=(6, 5))
        plt.imshow(attn_mean, aspect="auto", cmap="viridis")
        plt.colorbar(label="Attention weight")
        plt.title("Mean attention (tgt x src)")
        plt.xlabel("Source token")
        plt.ylabel("Target token")
        plt.tight_layout()
        plt.savefig(out_dir / "attention_heatmap.png", dpi=200)
        plt.close()

        print("Mean attention by modality:", modality_attn)

    # Optional: save a compact metrics file
    metrics_path = out_dir / "latent_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Linear probe R2: {r2}\n")
        f.write(f"Top dims (speed): {top_speed.tolist()}\n")
        f.write(f"Top dims (vz): {top_vz.tolist()}\n")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
