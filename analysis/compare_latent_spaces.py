import argparse
import json
import math
import sys
import warnings

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elope.datasets import ElopeDataLoader
from elope.models import build_model
from elope.utils import load_yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    HAS_MATPLOTLIB = False
    plt = None

try:
    from scipy.linalg import sqrtm
except Exception:  # pragma: no cover
    sqrtm = None


EPS = 1e-12


def _setup_plot_style() -> None:
    if not HAS_MATPLOTLIB:
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


def _to_optional_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    return float(s)


@dataclass
class LatentPack:
    name: str
    fused: np.ndarray
    pred: np.ndarray
    target_vel: np.ndarray
    target_pos: np.ndarray
    times: np.ndarray
    attention: np.ndarray | None = None
    event_tokens: np.ndarray | None = None
    total_tokens: np.ndarray | None = None
    flow_vector: np.ndarray | None = None
    event_density: np.ndarray | None = None
    sequence_id: np.ndarray | None = None
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def subset(self, idx: np.ndarray) -> "LatentPack":
        idx = np.asarray(idx, dtype=int)
        layers = {k: v[idx] for k, v in self.layers.items() if len(v) >= len(self.fused)}
        return LatentPack(
            name=self.name,
            fused=self.fused[idx],
            pred=self.pred[idx],
            target_vel=self.target_vel[idx],
            target_pos=self.target_pos[idx],
            times=self.times[idx],
            attention=self.attention[idx] if self.attention is not None and len(self.attention) >= len(self.fused) else self.attention,
            event_tokens=self.event_tokens[idx] if self.event_tokens is not None and len(self.event_tokens) >= len(self.fused) else self.event_tokens,
            total_tokens=self.total_tokens[idx] if self.total_tokens is not None and len(self.total_tokens) >= len(self.fused) else self.total_tokens,
            flow_vector=self.flow_vector[idx] if self.flow_vector is not None and len(self.flow_vector) >= len(self.fused) else self.flow_vector,
            event_density=self.event_density[idx] if self.event_density is not None and len(self.event_density) >= len(self.fused) else self.event_density,
            sequence_id=self.sequence_id[idx] if self.sequence_id is not None and len(self.sequence_id) >= len(self.fused) else self.sequence_id,
            layers=layers,
            meta=dict(self.meta),
        )


def _npz_get(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in npz.files:
        return None
    arr = npz[key]
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        if arr.item() is None:
            return None
    return arr


def _concat_nonempty(items: list[np.ndarray]) -> np.ndarray | None:
    items = [x for x in items if x is not None and len(x) > 0]
    if not items:
        return None
    return np.concatenate(items, axis=0)


def _sanitize_layer_key(name: str) -> str:
    return name.replace(".", "__")


def _unsanitize_layer_key(name: str) -> str:
    return name.replace("__", ".")


def load_latent_pack(path: Path, name: str) -> LatentPack:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found at {path}")

    fused_l, pred_l, tvel_l, tpos_l, times_l = [], [], [], [], []
    att_l, etok_l, ttok_l = [], [], []
    flow_vec_l, event_density_l = [], []
    layer_buffers: dict[str, list[np.ndarray]] = {}

    for f in files:
        npz = np.load(f, allow_pickle=True)
        fused = _npz_get(npz, "fused")
        pred = _npz_get(npz, "pred")
        tvel = _npz_get(npz, "target_vel")
        tpos = _npz_get(npz, "target_pos")
        times = _npz_get(npz, "times")
        if fused is None or pred is None or tvel is None or tpos is None or times is None:
            warnings.warn(f"Skipping malformed latent file: {f}")
            continue
        fused_l.append(fused)
        pred_l.append(pred)
        tvel_l.append(tvel)
        tpos_l.append(tpos)
        times_l.append(times.reshape(-1))
        att_l.append(_npz_get(npz, "attention"))
        etok_l.append(_npz_get(npz, "event_tokens"))
        ttok_l.append(_npz_get(npz, "total_tokens"))
        flow_vec_l.append(_npz_get(npz, "flow_vector"))
        event_density_l.append(_npz_get(npz, "event_density"))
        seq_ids = _npz_get(npz, "sequence_id")
        if seq_ids is not None:
            if seq_ids.dtype.kind in {"U", "S", "O"}:
                parsed = []
                for v in seq_ids.reshape(-1):
                    s = str(v)
                    parsed.append(int(s) if s.isdigit() else -1)
                seq_ids = np.asarray(parsed, dtype=np.int32)
            seq_ids = seq_ids.reshape(-1)
            layer_buffers.setdefault("__seq_id__", []).append(seq_ids)

        for key in npz.files:
            if key.startswith("layer__"):
                layer_name = _unsanitize_layer_key(key.replace("layer__", "", 1))
                layer_buffers.setdefault(layer_name, []).append(npz[key])

    fused = _concat_nonempty(fused_l)
    pred = _concat_nonempty(pred_l)
    target_vel = _concat_nonempty(tvel_l)
    target_pos = _concat_nonempty(tpos_l)
    times = _concat_nonempty(times_l)
    if fused is None or pred is None or target_vel is None or target_pos is None or times is None:
        raise RuntimeError(f"Could not load valid latent arrays from {path}")

    layers = {k: _concat_nonempty(v) for k, v in layer_buffers.items() if k != "__seq_id__"}
    layers = {k: v for k, v in layers.items() if v is not None}
    seq_id = _concat_nonempty(layer_buffers.get("__seq_id__", []))

    return LatentPack(
        name=name,
        fused=fused,
        pred=pred,
        target_vel=target_vel,
        target_pos=target_pos,
        times=times.reshape(-1),
        attention=_concat_nonempty(att_l),
        event_tokens=_concat_nonempty(etok_l),
        total_tokens=_concat_nonempty(ttok_l),
        flow_vector=_concat_nonempty(flow_vec_l),
        event_density=_concat_nonempty(event_density_l),
        sequence_id=seq_id,
        layers=layers,
        meta={"source": str(path), "files": [str(f) for f in files]},
    )


def _default_sequences() -> list[str]:
    return [f"{i:04d}" for i in range(28)]


def parse_sequences(s: str | None) -> list[str]:
    if s is None or s.strip() == "":
        return _default_sequences()
    out = []
    for token in s.split(","):
        token = token.strip()
        if token == "":
            continue
        if token.isdigit():
            out.append(f"{int(token):04d}")
        else:
            out.append(token)
    if not out:
        raise ValueError("No valid sequences parsed from --sequences.")
    return out


def _to_feature_2d(x: torch.Tensor, max_dim: int = 2048) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(1)
    elif x.ndim == 2:
        pass
    elif x.ndim == 3:
        x = x.mean(dim=1)
    else:
        x = x.reshape(x.shape[0], -1)
    if x.shape[1] > max_dim:
        x = x[:, :max_dim]
    return x


def _select_targets_times(
    output_type: str,
    seq_len: int,
    targets: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output_type == "initial_state":
        return targets[:, 0], times[:, 0]
    if output_type == "final_state":
        return targets[:, -1], times[:, -1]
    if output_type == "central_state":
        return targets[:, seq_len // 2], times[:, seq_len // 2]
    return targets[:, -1], times[:, -1]


def _select_prediction(output_type: str, seq_len: int, pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 2:
        return pred
    if pred.ndim != 3:
        return pred.reshape(pred.shape[0], -1)
    if output_type == "initial_state":
        return pred[:, 0]
    if output_type == "final_state":
        return pred[:, -1]
    if output_type == "central_state":
        return pred[:, seq_len // 2]
    return pred[:, -1]


def extract_latents_from_model(
    name: str,
    model_cfg_path: Path,
    dataset_cfg_path: Path,
    weights_path: Path,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    save_npz_path: Path | None = None,
) -> LatentPack:
    model_cfg = load_yaml(model_cfg_path)
    dataset_cfg = load_yaml(dataset_cfg_path)

    event_encoder = dataset_cfg["events"]["encoder_method"]
    event_norm = str(model_cfg["event_normalization"])
    event_integration_window = _to_optional_float(model_cfg.get("event_integration_window", None))
    if event_integration_window is not None and event_encoder != "last_timestamp":
        event_integration_window = None

    sample_len = int(model_cfg["sequence_length"])
    sample_interval = int(model_cfg.get("sample_interval", 1))
    loader = ElopeDataLoader(
        dataset_cfg_path,
        sequence_ids=sequences,
        sample_len=sample_len,
        sample_interval=sample_interval,
        padding=str(model_cfg["padding"]),
        event_normalization=event_norm,
        event_integration_window=event_integration_window,
        augment=False,
        flip=0.0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dataset = loader.dataset
    seq_id_map = []
    for sid, slen in zip(dataset.seq_ids, dataset.seq_lengths):
        sid_i = int(sid) if str(sid).isdigit() else -1
        seq_id_map.extend([sid_i] * int(slen))
    seq_id_map = np.asarray(seq_id_map, dtype=np.int32)

    model = build_model(model_cfg, dataset_cfg, device=device)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    layer_names = []
    for candidate in ["encoder_event", "encoder_angle", "encoder_omega", "encoder_range", "fusion", "regressor"]:
        if hasattr(model, candidate):
            layer_names.append(candidate)

    layer_cache: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_name: str):
        def _hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(out):
                layer_cache[layer_name] = _to_feature_2d(out.detach())
        return _hook

    for ln in layer_names:
        hooks.append(getattr(model, ln).register_forward_hook(make_hook(ln)))

    fused_l, pred_l, tvel_l, tpos_l, times_l = [], [], [], [], []
    att_l, etok_l, ttok_l = [], [], []
    flow_vec_l, event_density_l = [], []
    seq_id_l = []
    layer_buffers: dict[str, list[np.ndarray]] = {k: [] for k in layer_names}

    output_type = str(model_cfg["output_type"])
    seq_len = int(model_cfg["sequence_length"])

    cursor = 0
    with torch.no_grad():
        for bidx, (events, imus, ranges, targets, times) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            bsz = int(events.shape[0])
            if cursor + bsz <= len(seq_id_map):
                seq_id_l.append(seq_id_map[cursor:cursor + bsz])
            else:
                seq_id_l.append(np.full(bsz, -1, dtype=np.int32))
            cursor += bsz

            layer_cache.clear()
            events = events.to(device)
            imus = imus.to(device)
            ranges = ranges.to(device)
            targets = targets.to(device)
            times = times.to(device)

            outputs = model(times, events, imus, ranges)
            pred = _select_prediction(output_type, seq_len, outputs["prediction"])

            tgt_sel, tms_sel = _select_targets_times(output_type, seq_len, targets, times)
            tpos = tgt_sel[:, 0:3]
            tvel = tgt_sel[:, 3:6]

            fused = layer_cache.get("fusion", None)
            if fused is None:
                raise RuntimeError("Could not capture `fusion` layer output; model may not expose `fusion`.")

            fused_l.append(fused.cpu().numpy())
            pred_l.append(pred.detach().cpu().numpy())
            tvel_l.append(tvel.detach().cpu().numpy())
            tpos_l.append(tpos.detach().cpu().numpy())
            times_l.append(tms_sel.detach().cpu().numpy().reshape(-1))

            att = outputs.get("attention_weights", None)
            att_l.append(att.detach().cpu().numpy() if torch.is_tensor(att) else None)

            event_tokens = torch.tensor([events.shape[-3] if events.ndim >= 6 else -1] * events.shape[0])
            total_tokens = event_tokens + 3
            etok_l.append(event_tokens.numpy())
            ttok_l.append(total_tokens.numpy())

            flow = outputs.get("flow_prediction", None)
            if torch.is_tensor(flow):
                if flow.ndim == 4:
                    flow_vec = flow.mean(dim=(2, 3))
                else:
                    flow_vec = flow.reshape(flow.shape[0], -1)
                    if flow_vec.shape[1] > 2:
                        flow_vec = flow_vec[:, :2]
                flow_vec_l.append(flow_vec.detach().cpu().numpy())
            else:
                flow_vec_l.append(None)

            ev_density = events[:, -1].abs().mean(dim=(1, 2, 3, 4)).detach().cpu().numpy()
            event_density_l.append(ev_density)

            for ln in layer_names:
                if ln in layer_cache:
                    layer_buffers[ln].append(layer_cache[ln].cpu().numpy())

    for h in hooks:
        h.remove()

    fused = _concat_nonempty(fused_l)
    pred = _concat_nonempty(pred_l)
    target_vel = _concat_nonempty(tvel_l)
    target_pos = _concat_nonempty(tpos_l)
    times = _concat_nonempty(times_l)
    if fused is None or pred is None or target_vel is None or target_pos is None or times is None:
        raise RuntimeError(f"No samples extracted for {name}.")

    layers = {k: _concat_nonempty(v) for k, v in layer_buffers.items()}
    layers = {k: v for k, v in layers.items() if v is not None}

    pack = LatentPack(
        name=name,
        fused=fused,
        pred=pred,
        target_vel=target_vel,
        target_pos=target_pos,
        times=times.reshape(-1),
        attention=_concat_nonempty(att_l),
        event_tokens=_concat_nonempty(etok_l),
        total_tokens=_concat_nonempty(ttok_l),
        flow_vector=_concat_nonempty(flow_vec_l),
        event_density=_concat_nonempty(event_density_l),
        sequence_id=_concat_nonempty(seq_id_l),
        layers=layers,
        meta={
            "source": "model_inference",
            "model_cfg": str(model_cfg_path),
            "dataset_cfg": str(dataset_cfg_path),
            "weights": str(weights_path),
            "sequences": sequences,
            "max_batches": max_batches,
        },
    )

    if save_npz_path is not None:
        save_npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fused": pack.fused,
            "pred": pack.pred,
            "target_vel": pack.target_vel,
            "target_pos": pack.target_pos,
            "times": pack.times,
            "attention": pack.attention,
            "event_tokens": pack.event_tokens,
            "total_tokens": pack.total_tokens,
            "flow_vector": pack.flow_vector,
            "event_density": pack.event_density,
            "sequence_id": pack.sequence_id,
        }
        for ln, arr in pack.layers.items():
            payload[f"layer__{_sanitize_layer_key(ln)}"] = arr
        np.savez_compressed(save_npz_path, **payload)

    return pack


def _build_sample_keys(pack: LatentPack, decimals: int = 4) -> list[tuple[float, ...]]:
    t = pack.times.reshape(-1, 1)
    v = pack.target_vel.reshape(len(pack.fused), -1)
    p = pack.target_pos.reshape(len(pack.fused), -1)
    mat = np.concatenate([t, p, v], axis=1)
    mat = np.round(mat, decimals=decimals)
    return [tuple(row.tolist()) for row in mat]


def align_packs(pack_a: LatentPack, pack_b: LatentPack, decimals: int = 4) -> tuple[LatentPack, LatentPack, dict]:
    keys_a = _build_sample_keys(pack_a, decimals=decimals)
    keys_b = _build_sample_keys(pack_b, decimals=decimals)

    buckets: dict[tuple[float, ...], list[int]] = {}
    for i, key in enumerate(keys_b):
        buckets.setdefault(key, []).append(i)

    idx_a = []
    idx_b = []
    for i, key in enumerate(keys_a):
        lst = buckets.get(key, None)
        if lst:
            idx_a.append(i)
            idx_b.append(lst.pop())

    stats = {
        "matched": len(idx_a),
        "a_total": len(keys_a),
        "b_total": len(keys_b),
        "match_ratio_a": float(len(idx_a) / max(len(keys_a), 1)),
        "match_ratio_b": float(len(idx_a) / max(len(keys_b), 1)),
        "mode": "key_match",
    }

    if len(idx_a) < max(200, int(0.4 * min(len(keys_a), len(keys_b)))):
        n = min(len(keys_a), len(keys_b))
        idx_a = np.arange(n)
        idx_b = np.arange(n)
        stats["mode"] = "fallback_order"
        stats["matched"] = int(n)
        stats["match_ratio_a"] = float(n / max(len(keys_a), 1))
        stats["match_ratio_b"] = float(n / max(len(keys_b), 1))

    idx_a = np.asarray(idx_a, dtype=int)
    idx_b = np.asarray(idx_b, dtype=int)
    return pack_a.subset(idx_a), pack_b.subset(idx_b), stats


def _train_test_split(n: int, seed: int, frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = max(1, int(frac * n))
    return idx[:ntr], idx[ntr:]


def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma < EPS, 1.0, sigma)
    return mu, sigma


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _ridge_regression(X: np.ndarray, Y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    I = np.eye(X1.shape[1], dtype=X1.dtype)
    I[-1, -1] = 0.0
    A = X1.T @ X1 + l2 * I
    B = X1.T @ Y
    return np.linalg.solve(A, B)


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return X1 @ W


def _regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, EPS)
    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(Y_true - Y_pred), axis=0)
    return {
        "r2_vx": float(r2[0]),
        "r2_vy": float(r2[1]),
        "r2_vz": float(r2[2]),
        "r2_mean": float(np.mean(r2)),
        "rmse_vx": float(rmse[0]),
        "rmse_vy": float(rmse[1]),
        "rmse_vz": float(rmse[2]),
        "rmse_mean": float(np.mean(rmse)),
        "mae_vx": float(mae[0]),
        "mae_vy": float(mae[1]),
        "mae_vz": float(mae[2]),
        "mae_mean": float(np.mean(mae)),
    }


def _onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> float:
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        if p + r == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else float("nan")


def _linear_probe_classification(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    l2: float = 1e-3,
) -> dict[str, float]:
    valid = y >= 0
    X = X[valid]
    y = y[valid]
    if len(y) < 30:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "n": int(len(y)),
            "classes": int(len(np.unique(y))) if len(y) > 0 else 0,
            "degenerate": True,
        }

    classes = np.unique(y)
    if len(classes) < 2:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "n": int(len(y)),
            "classes": int(len(classes)),
            "degenerate": True,
        }
    class_to_id = {c: i for i, c in enumerate(classes)}
    y_id = np.array([class_to_id[c] for c in y], dtype=int)
    train_idx, test_idx = _train_test_split(len(y_id), seed=seed, frac=0.8)
    if len(test_idx) == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "n": int(len(y)),
            "classes": int(len(classes)),
            "degenerate": True,
        }

    mu, sigma = _zscore_fit(X[train_idx])
    Xtr = _zscore_apply(X[train_idx], mu, sigma)
    Xte = _zscore_apply(X[test_idx], mu, sigma)

    Ytr = _onehot(y_id[train_idx], num_classes=len(classes))
    W = _ridge_regression(Xtr, Ytr, l2=l2)
    logits = _ridge_predict(Xte, W)
    yhat = np.argmax(logits, axis=1)
    yt = y_id[test_idx]

    return {
        "accuracy": float(np.mean(yhat == yt)),
        "macro_f1": _macro_f1(yt, yhat, np.arange(len(classes))),
        "n": int(len(y)),
        "classes": int(len(classes)),
        "degenerate": False,
    }


def _speed_bins(speed: np.ndarray, n_bins: int = 4) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(speed, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(speed, dtype=int)
    bins = np.digitize(speed, edges[1:-1], right=False)
    return bins.astype(int)


def _direction_bins(v: np.ndarray, speed: np.ndarray, n_bins: int = 8, static_q: float = 0.2) -> np.ndarray:
    ang = np.arctan2(v[:, 1], v[:, 0])
    ang = (ang + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(ang / (2 * np.pi / n_bins)).astype(int)
    static_thr = np.quantile(speed, static_q)
    bins[speed <= static_thr] = -1
    return bins


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.maximum(sq + sq.T - 2 * G, 0.0)
    return np.sqrt(D2 + EPS)


def _intrinsic_dim_levina_bickel(X: np.ndarray, k: int = 10, max_points: int = 1500, seed: int = 0) -> float:
    if len(X) < k + 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        idx = rng.choice(len(X), size=max_points, replace=False)
        X = X[idx]
    D = _pairwise_dist(X)
    np.fill_diagonal(D, np.inf)
    D_sorted = np.sort(D, axis=1)
    rk = D_sorted[:, k - 1]
    rj = D_sorted[:, :k - 1]
    valid = np.all(rj > 0, axis=1) & (rk > 0)
    if not np.any(valid):
        return float("nan")
    logs = np.log(np.maximum(rk[valid, None] / np.maximum(rj[valid], EPS), 1.0 + EPS))
    m = np.mean(logs, axis=1)
    inv_m = 1.0 / np.maximum(m, EPS)
    return float(np.mean(inv_m))


def _fisher_ratio(X: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    X = X[valid]
    y = labels[valid]
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")

    mu = X.mean(axis=0, keepdims=True)
    sb = 0.0
    sw = 0.0
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 2:
            continue
        muc = Xc.mean(axis=0, keepdims=True)
        sb += len(Xc) * float(np.sum((muc - mu) ** 2))
        sw += float(np.sum((Xc - muc) ** 2))
    if sw <= 0:
        return float("nan")
    return float(sb / sw)


def _silhouette_score(X: np.ndarray, labels: np.ndarray, max_points: int = 1200, seed: int = 0) -> float:
    valid = labels >= 0
    X = X[valid]
    y = labels[valid]
    if len(X) < 20:
        return float("nan")
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        idx = rng.choice(len(X), size=max_points, replace=False)
        X = X[idx]
        y = y[idx]

    D = _pairwise_dist(X)
    svals = []
    for i in range(len(X)):
        yi = y[i]
        same = y == yi
        same[i] = False
        if np.any(same):
            a = float(np.mean(D[i, same]))
        else:
            a = 0.0
        b = math.inf
        for c in classes:
            if c == yi:
                continue
            mask = y == c
            if np.any(mask):
                b = min(b, float(np.mean(D[i, mask])))
        if not math.isfinite(b):
            continue
        svals.append((b - a) / max(a, b, EPS))
    if not svals:
        return float("nan")
    return float(np.mean(svals))


def _compute_temporal_features(pack: LatentPack) -> dict[str, Any]:
    times = pack.times.reshape(-1)
    v = pack.target_vel
    z = pack.fused

    boundaries = [0]
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            boundaries.append(i)
    boundaries.append(len(times))

    segments = []
    accel = np.full(len(times), np.nan, dtype=np.float64)
    turn_rate = np.full(len(times), np.nan, dtype=np.float64)
    smoothness_vals = []
    short_long_vals = []
    adj_vals = []
    lag_vals = []

    def angle_wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    lag = 5
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s < 6:
            continue
        t = times[s:e]
        vv = v[s:e]
        zz = z[s:e]
        dt = np.diff(t)
        good = dt > EPS
        if np.sum(good) < 4:
            continue
        segments.append((s, e))

        dv = vv[1:] - vv[:-1]
        a = np.full(len(vv), np.nan)
        a[1:] = np.linalg.norm(dv, axis=1) / np.maximum(dt, EPS)
        accel[s:e] = a

        ang = np.arctan2(vv[:, 1], vv[:, 0])
        dth = np.full(len(vv), np.nan)
        dth[1:] = np.abs(angle_wrap(ang[1:] - ang[:-1])) / np.maximum(dt, EPS)
        turn_rate[s:e] = dth

        d1 = np.linalg.norm(zz[1:] - zz[:-1], axis=1)
        adj_vals.append(float(np.mean(d1)))
        if len(zz) > lag:
            dlag = np.linalg.norm(zz[lag:] - zz[:-lag], axis=1)
            lag_vals.append(float(np.mean(dlag)))
            short_long_vals.append(float(np.mean(d1) / max(np.mean(dlag), EPS)))
        if len(zz) > 2:
            s2 = np.linalg.norm(zz[2:] - 2 * zz[1:-1] + zz[:-2], axis=1)
            smoothness_vals.append(float(np.mean(s2)))

    valid_temporal = len(segments) > 0
    return {
        "segments": segments,
        "num_segments": len(segments),
        "valid_temporal": valid_temporal,
        "accel": accel,
        "turn_rate": turn_rate,
        "adjacent_distance_mean": float(np.mean(adj_vals)) if adj_vals else float("nan"),
        "lag_distance_mean": float(np.mean(lag_vals)) if lag_vals else float("nan"),
        "short_long_ratio": float(np.mean(short_long_vals)) if short_long_vals else float("nan"),
        "latent_smoothness": float(np.mean(smoothness_vals)) if smoothness_vals else float("nan"),
    }


def _cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    nx = np.linalg.norm(Xc.T @ Xc, ord="fro")
    ny = np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(hsic / max(nx * ny, EPS))


def _pca_reduce(X: np.ndarray, keep_var: float = 0.99) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(X) - 1, 1)
    csum = np.cumsum(var) / max(np.sum(var), EPS)
    k = int(np.searchsorted(csum, keep_var) + 1)
    k = max(1, min(k, U.shape[1]))
    return U[:, :k] * S[:k]


def _svcca_similarity(X: np.ndarray, Y: np.ndarray, keep_var: float = 0.99) -> float:
    Xr = _pca_reduce(X, keep_var=keep_var)
    Yr = _pca_reduce(Y, keep_var=keep_var)
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
    Cxx_inv_half = vx @ np.diag(1.0 / np.sqrt(wx)) @ vx.T
    Cyy_inv_half = vy @ np.diag(1.0 / np.sqrt(wy)) @ vy.T

    M = Cxx_inv_half @ Cxy @ Cyy_inv_half
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))


def _cov_eigs(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 0.0, None)
    return np.sort(w)[::-1]


def _participation_ratio(eigs: np.ndarray) -> float:
    s1 = float(np.sum(eigs))
    s2 = float(np.sum(eigs ** 2))
    if s2 <= 0:
        return float("nan")
    return float((s1 ** 2) / s2)


def _effective_rank(eigs: np.ndarray) -> float:
    s = eigs / max(np.sum(eigs), EPS)
    s = np.clip(s, EPS, None)
    return float(np.exp(-np.sum(s * np.log(s))))


def _isotropy(eigs: np.ndarray) -> float:
    pos = eigs[eigs > 1e-10]
    if len(pos) == 0:
        return float("nan")
    return float(np.min(pos) / np.max(pos))


def _top_explained(eigs: np.ndarray, k: int = 3) -> list[float]:
    s = eigs / max(np.sum(eigs), EPS)
    k = min(k, len(s))
    return [float(v) for v in s[:k]]


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, max_points: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        X = X[rng.choice(len(X), size=max_points, replace=False)]
    if len(Y) > max_points:
        Y = Y[rng.choice(len(Y), size=max_points, replace=False)]

    Z = np.concatenate([X, Y], axis=0)
    D = _pairwise_dist(Z)
    tri = D[np.triu_indices_from(D, k=1)]
    sigma = np.median(tri)
    sigma = max(float(sigma), 1e-3)
    gamma = 1.0 / (2 * sigma * sigma)

    def rbf(A, B):
        sqA = np.sum(A * A, axis=1, keepdims=True)
        sqB = np.sum(B * B, axis=1, keepdims=True).T
        D2 = np.maximum(sqA + sqB - 2 * (A @ B.T), 0.0)
        return np.exp(-gamma * D2)

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    m = len(X)
    n = len(Y)
    term_xx = np.sum(Kxx) / max(m * (m - 1), 1)
    term_yy = np.sum(Kyy) / max(n * (n - 1), 1)
    term_xy = np.sum(Kxy) / max(m * n, 1)
    return float(term_xx + term_yy - 2 * term_xy)


def _frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    mu1 = X.mean(axis=0)
    mu2 = Y.mean(axis=0)
    C1 = np.cov(X, rowvar=False)
    C2 = np.cov(Y, rowvar=False)
    delta = mu1 - mu2

    if sqrtm is None:
        return float("nan")

    covmean = sqrtm(C1 @ C2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    val = float(delta @ delta + np.trace(C1 + C2 - 2.0 * covmean))
    return val


def _latent_target_corr(X: np.ndarray, target_vel: np.ndarray, max_dims: int) -> np.ndarray:
    speed = np.linalg.norm(target_vel, axis=1)
    D = min(max_dims, X.shape[1])
    mat = np.concatenate(
        [target_vel[:, :3], speed[:, None], X[:, :D]],
        axis=1,
    )
    corr = np.corrcoef(mat, rowvar=False)
    return corr[:4, 4:]


def _pca_project(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    n = max(len(Xc) - 1, 1)
    var = (S ** 2) / n
    var_ratio = var / max(np.sum(var), EPS)
    Z = U[:, :n_components] * S[:n_components]
    return Z, var_ratio


def _plot_pca_compare(pack_a: LatentPack, pack_b: LatentPack, out_path: Path, color_mode: str = "speed") -> None:
    if not HAS_MATPLOTLIB:
        return
    X = np.concatenate([pack_a.fused, pack_b.fused], axis=0)
    Z3, var_ratio = _pca_project(X, n_components=3)
    Z = Z3[:, :2]
    Za = Z[:len(pack_a.fused)]
    Zb = Z[len(pack_a.fused):]

    if color_mode == "vz":
        ca = pack_a.target_vel[:, 2]
        cb = pack_b.target_vel[:, 2]
        cbar_title = "vz"
    else:
        ca = np.linalg.norm(pack_a.target_vel, axis=1)
        cb = np.linalg.norm(pack_b.target_vel, axis=1)
        cbar_title = "speed"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True, constrained_layout=True)
    im0 = axes[0].scatter(Za[:, 0], Za[:, 1], c=ca, s=8, cmap="viridis", alpha=0.78, linewidths=0)
    axes[0].set_title(f"{pack_a.name} (PC1+PC2 var={100*float(np.sum(var_ratio[:2])):.1f}%)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, linestyle=":", linewidth=0.7)
    im1 = axes[1].scatter(Zb[:, 0], Zb[:, 1], c=cb, s=8, cmap="viridis", alpha=0.78, linewidths=0)
    axes[1].set_title(f"{pack_b.name} (joint PCA)")
    axes[1].set_xlabel("PC1")
    axes[1].grid(True, linestyle=":", linewidth=0.7)
    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_pca_3d_compare(pack_a: LatentPack, pack_b: LatentPack, out_path: Path, color_mode: str = "speed") -> None:
    if not HAS_MATPLOTLIB:
        return
    X = np.concatenate([pack_a.fused, pack_b.fused], axis=0)
    Z, var_ratio = _pca_project(X, n_components=3)
    Za = Z[:len(pack_a.fused)]
    Zb = Z[len(pack_a.fused):]

    if color_mode == "vz":
        ca = pack_a.target_vel[:, 2]
        cb = pack_b.target_vel[:, 2]
        cbar_title = "vz"
    else:
        ca = np.linalg.norm(pack_a.target_vel, axis=1)
        cb = np.linalg.norm(pack_b.target_vel, axis=1)
        cbar_title = "speed"

    fig = plt.figure(figsize=(12.8, 5.8), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    im0 = ax0.scatter(Za[:, 0], Za[:, 1], Za[:, 2], c=ca, s=8, cmap="viridis", alpha=0.75, linewidths=0)
    ax0.set_title(pack_a.name)
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.set_zlabel("PC3")
    im1 = ax1.scatter(Zb[:, 0], Zb[:, 1], Zb[:, 2], c=cb, s=8, cmap="viridis", alpha=0.75, linewidths=0)
    ax1.set_title(pack_b.name)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    cbar = fig.colorbar(im1, ax=[ax0, ax1], fraction=0.03, pad=0.03)
    cbar.set_label(cbar_title)
    fig.suptitle(
        "Joint PCA 3D "
        f"(PC1={100*var_ratio[0]:.1f}%, PC2={100*var_ratio[1]:.1f}%, PC3={100*var_ratio[2]:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_corr_heatmap(corr: np.ndarray, out_path: Path, title: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    im = ax.imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")
    ax.set_yticks([0, 1, 2, 3], ["vx", "vy", "vz", "speed"])
    ax.set_xlabel("Latent dim")
    ax.set_title(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_eigenspectrum(eigs_a: np.ndarray, eigs_b: np.ndarray, name_a: str, name_b: str, out_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.figure(figsize=(8.5, 5.2), constrained_layout=True)
    xa = np.arange(1, len(eigs_a) + 1)
    xb = np.arange(1, len(eigs_b) + 1)
    plt.semilogy(xa, eigs_a / max(np.sum(eigs_a), EPS), label=name_a, color="#0f766e", linewidth=2.1)
    plt.semilogy(xb, eigs_b / max(np.sum(eigs_b), EPS), label=name_b, color="#b45309", linewidth=2.1)
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Normalized eigenvalue (log)")
    plt.title("Latent covariance eigenspectrum")
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_layerwise_similarity(layer_metrics: dict[str, dict[str, float]], out_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    layers = list(layer_metrics.keys())
    cka_vals = [layer_metrics[k]["cka"] for k in layers]
    svcca_vals = [layer_metrics[k]["svcca"] for k in layers]
    x = np.arange(len(layers))
    w = 0.38
    plt.figure(figsize=(max(8.5, len(layers) * 1.15), 5.2), constrained_layout=True)
    plt.bar(x - w / 2, cka_vals, width=w, label="CKA", color="#0f766e")
    plt.bar(x + w / 2, svcca_vals, width=w, label="SVCCA", color="#b45309")
    plt.ylim(0, 1.05)
    plt.xticks(x, layers, rotation=30, ha="right")
    plt.ylabel("Similarity")
    plt.title("Layerwise representation similarity")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_probe_scores(probes_a: dict, probes_b: dict, name_a: str, name_b: str, out_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    labels = [
        "vel_r2_mean",
        "speed_bin_acc",
        "dir_bin_acc",
        "static_dyn_acc",
        "motion_boundary_acc",
    ]
    va = [
        probes_a["velocity_regression"]["r2_mean"],
        probes_a["speed_bin_probe"]["accuracy"],
        probes_a["direction_probe"]["accuracy"],
        probes_a["static_dynamic_probe"]["accuracy"],
        probes_a["motion_boundary_probe"]["accuracy"],
    ]
    vb = [
        probes_b["velocity_regression"]["r2_mean"],
        probes_b["speed_bin_probe"]["accuracy"],
        probes_b["direction_probe"]["accuracy"],
        probes_b["static_dynamic_probe"]["accuracy"],
        probes_b["motion_boundary_probe"]["accuracy"],
    ]
    va_plot = np.nan_to_num(np.asarray(va, dtype=float), nan=0.0)
    vb_plot = np.nan_to_num(np.asarray(vb, dtype=float), nan=0.0)
    x = np.arange(len(labels))
    w = 0.38
    plt.figure(figsize=(10.5, 5.2), constrained_layout=True)
    plt.bar(x - w / 2, va_plot, width=w, label=name_a, color="#0f766e")
    plt.bar(x + w / 2, vb_plot, width=w, label=name_b, color="#b45309")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Linear probe comparison")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_robustness_slices(
    slice_a: dict[str, dict[str, float]],
    slice_b: dict[str, dict[str, float]],
    name_a: str,
    name_b: str,
    out_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    common = [k for k in slice_a.keys() if k in slice_b and slice_a[k]["n"] > 20 and slice_b[k]["n"] > 20]
    if not common:
        return
    vals_a = [slice_a[k]["rmse"] for k in common]
    vals_b = [slice_b[k]["rmse"] for k in common]
    x = np.arange(len(common))
    w = 0.38
    plt.figure(figsize=(max(9, len(common) * 1.25), 5.2), constrained_layout=True)
    plt.bar(x - w / 2, vals_a, width=w, label=name_a, color="#0f766e")
    plt.bar(x + w / 2, vals_b, width=w, label=name_b, color="#b45309")
    plt.xticks(x, common, rotation=30, ha="right")
    plt.ylabel("Velocity RMSE")
    plt.title("Robustness by motion/data slice")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_attention(att: np.ndarray, out_path: Path, title: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    if att.ndim == 4:
        att_mean = att.mean(axis=(0, 1))
    elif att.ndim == 3:
        att_mean = att.mean(axis=0)
    else:
        att_mean = att
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    im = ax.imshow(att_mean, aspect="auto", cmap="magma")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Attention weight")
    ax.set_xlabel("Source token")
    ax.set_ylabel("Target token")
    ax.set_title(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _attention_modality(attention: np.ndarray | None, event_tokens: np.ndarray | None) -> dict[str, float]:
    if attention is None:
        return {}
    if attention.ndim == 4:
        att = attention.mean(axis=(0, 1))
    elif attention.ndim == 3:
        att = attention.mean(axis=0)
    else:
        att = attention
    if event_tokens is not None and len(event_tokens) > 0:
        e_end = int(np.median(event_tokens[event_tokens > 0])) if np.any(event_tokens > 0) else 4
    else:
        e_end = max(att.shape[1] - 3, 1)
    imu_idx = e_end
    range_idx = e_end + 1
    angle_idx = e_end + 2
    out = {}
    out["event"] = float(att[:, :e_end].mean()) if e_end > 0 else float("nan")
    out["imu"] = float(att[:, imu_idx].mean()) if imu_idx < att.shape[1] else float("nan")
    out["range"] = float(att[:, range_idx].mean()) if range_idx < att.shape[1] else float("nan")
    out["angle"] = float(att[:, angle_idx].mean()) if angle_idx < att.shape[1] else float("nan")
    return out


def _knn_purity(X: np.ndarray, labels: np.ndarray, k: int = 10, max_points: int = 2000, seed: int = 0) -> float:
    valid = labels >= 0
    X = X[valid]
    y = labels[valid]
    if len(X) < k + 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        idx = rng.choice(len(X), size=max_points, replace=False)
        X = X[idx]
        y = y[idx]
    D = _pairwise_dist(X)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=k, axis=1)[:, :k]
    same = (y[nn] == y[:, None]).mean(axis=1)
    return float(np.mean(same))


def analyze_pack(pack: LatentPack, seed: int, max_corr_dims: int) -> dict[str, Any]:
    n = len(pack.fused)
    speed = np.linalg.norm(pack.target_vel, axis=1)
    pred_err = np.linalg.norm(pack.pred - pack.target_vel, axis=1)

    train_idx, test_idx = _train_test_split(n, seed=seed, frac=0.8)
    mu, sigma = _zscore_fit(pack.fused[train_idx])
    Xtr = _zscore_apply(pack.fused[train_idx], mu, sigma)
    Xte = _zscore_apply(pack.fused[test_idx], mu, sigma)

    W = _ridge_regression(Xtr, pack.target_vel[train_idx], l2=1e-3)
    yhat = _ridge_predict(Xte, W)
    vel_probe = _regression_metrics(pack.target_vel[test_idx], yhat)

    speed_bins = _speed_bins(speed, n_bins=4)
    dir_bins = _direction_bins(pack.target_vel, speed, n_bins=8, static_q=0.2)
    static_dyn = (speed > np.quantile(speed, 0.25)).astype(int)

    temporal = _compute_temporal_features(pack)
    accel = temporal["accel"]
    if np.any(np.isfinite(accel)):
        thr = np.nanquantile(accel, 0.75)
        motion_boundary = np.where(np.isfinite(accel), (accel >= thr).astype(int), -1)
    else:
        motion_boundary = np.full(n, -1, dtype=int)

    probes = {
        "velocity_regression": vel_probe,
        "speed_bin_probe": _linear_probe_classification(pack.fused, speed_bins, seed=seed + 11),
        "direction_probe": _linear_probe_classification(pack.fused, dir_bins, seed=seed + 12),
        "static_dynamic_probe": _linear_probe_classification(pack.fused, static_dyn, seed=seed + 13),
        "motion_boundary_probe": _linear_probe_classification(pack.fused, motion_boundary, seed=seed + 14),
    }

    eigs = _cov_eigs(pack.fused)
    geometry = {
        "participation_ratio": _participation_ratio(eigs),
        "effective_rank": _effective_rank(eigs),
        "isotropy": _isotropy(eigs),
        "intrinsic_dim_lb": _intrinsic_dim_levina_bickel(pack.fused, k=10, max_points=1500, seed=seed),
        "pca_dim_90": int(np.searchsorted(np.cumsum(eigs) / max(np.sum(eigs), EPS), 0.90) + 1),
        "pca_dim_95": int(np.searchsorted(np.cumsum(eigs) / max(np.sum(eigs), EPS), 0.95) + 1),
        "fisher_speed_bin": _fisher_ratio(pack.fused, speed_bins),
        "fisher_direction": _fisher_ratio(pack.fused, dir_bins),
        "silhouette_speed_bin": _silhouette_score(pack.fused, speed_bins, max_points=1000, seed=seed),
        "silhouette_direction": _silhouette_score(pack.fused, dir_bins, max_points=1000, seed=seed + 1),
        "explained_var_top3": _top_explained(eigs, k=3),
    }

    knn = {
        "purity_speed_bin_k10": _knn_purity(pack.fused, speed_bins, k=10, seed=seed),
        "purity_direction_k10": _knn_purity(pack.fused, dir_bins, k=10, seed=seed + 1),
        "purity_static_dynamic_k10": _knn_purity(pack.fused, static_dyn, k=10, seed=seed + 2),
    }

    slices: dict[str, np.ndarray] = {}
    q25 = np.quantile(speed, 0.25)
    q75 = np.quantile(speed, 0.75)
    vxvy = np.linalg.norm(pack.target_vel[:, :2], axis=1)
    q75_xy = np.quantile(vxvy, 0.75)
    q75_vz = np.quantile(np.abs(pack.target_vel[:, 2]), 0.75)

    slices["all"] = np.ones(n, dtype=bool)
    slices["low_speed"] = speed <= q25
    slices["high_speed"] = speed >= q75
    slices["lateral_fast"] = vxvy >= q75_xy
    slices["vertical_fast"] = np.abs(pack.target_vel[:, 2]) >= q75_vz
    slices["static"] = speed <= np.quantile(speed, 0.1)

    if np.any(np.isfinite(accel)):
        slices["high_accel"] = np.isfinite(accel) & (accel >= np.nanquantile(accel, 0.75))
    if np.any(np.isfinite(temporal["turn_rate"])):
        turn = temporal["turn_rate"]
        slices["high_turn_rate"] = np.isfinite(turn) & (turn >= np.nanquantile(turn, 0.75))
    if pack.event_density is not None:
        ed = pack.event_density.reshape(-1)
        slices["sparse_events"] = ed <= np.quantile(ed, 0.25)
        slices["dense_events"] = ed >= np.quantile(ed, 0.75)

    slice_metrics = {}
    for k, mask in slices.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        e = pred_err[mask]
        slice_metrics[k] = {
            "n": int(mask.sum()),
            "frac": float(mask.mean()),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(e)),
            "median_err": float(np.median(e)),
        }

    sequence_metrics = {}
    if pack.sequence_id is not None:
        sid = pack.sequence_id.reshape(-1).astype(int)
        for s in np.unique(sid):
            if s < 0:
                continue
            mask = sid == s
            if np.sum(mask) < 20:
                continue
            e = pred_err[mask]
            sequence_metrics[f"{s:04d}"] = {
                "n": int(np.sum(mask)),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "mae": float(np.mean(e)),
                "median_err": float(np.median(e)),
                "speed_mean": float(np.mean(speed[mask])),
            }

    basic = {
        "n_samples": int(n),
        "latent_dim": int(pack.fused.shape[1]),
        "speed_mean": float(speed.mean()),
        "speed_std": float(speed.std()),
        "pred_rmse": float(np.sqrt(np.mean(pred_err ** 2))),
        "pred_mae": float(np.mean(pred_err)),
    }

    corr = _latent_target_corr(pack.fused, pack.target_vel, max_dims=max_corr_dims)
    attention_mod = _attention_modality(pack.attention, pack.event_tokens)

    return {
        "basic": basic,
        "probes": probes,
        "geometry": geometry,
        "temporal": {
            "valid_temporal": bool(temporal["valid_temporal"]),
            "num_segments": int(temporal["num_segments"]),
            "adjacent_distance_mean": float(temporal["adjacent_distance_mean"]),
            "lag_distance_mean": float(temporal["lag_distance_mean"]),
            "short_long_ratio": float(temporal["short_long_ratio"]),
            "latent_smoothness": float(temporal["latent_smoothness"]),
        },
        "knn": knn,
        "slice_metrics": slice_metrics,
        "sequence_metrics": sequence_metrics,
        "attention_modality": attention_mod,
        "corr": corr,
        "eigs": eigs,
        "labels": {
            "speed_bins": speed_bins,
            "direction_bins": dir_bins,
            "static_dynamic": static_dyn,
            "motion_boundary": motion_boundary,
        },
    }


def compare_packs(
    pack_flow: LatentPack,
    pack_noflow: LatentPack,
    analysis_flow: dict[str, Any],
    analysis_noflow: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    layer_metrics: dict[str, dict[str, float]] = {}

    layers_flow = dict(pack_flow.layers)
    layers_noflow = dict(pack_noflow.layers)
    layers_flow["fused"] = pack_flow.fused
    layers_noflow["fused"] = pack_noflow.fused
    common_layers = sorted(set(layers_flow.keys()).intersection(layers_noflow.keys()))

    rng = np.random.default_rng(seed)
    for ln in common_layers:
        Xa = layers_flow[ln]
        Xb = layers_noflow[ln]
        n = min(len(Xa), len(Xb))
        Xa = Xa[:n]
        Xb = Xb[:n]
        if n > 3000:
            idx = rng.choice(n, size=3000, replace=False)
            Xa = Xa[idx]
            Xb = Xb[idx]
        layer_metrics[ln] = {
            "cka": _cka_linear(Xa, Xb),
            "svcca": _svcca_similarity(Xa, Xb),
            "n": int(len(Xa)),
            "dim_flow": int(Xa.shape[1]),
            "dim_noflow": int(Xb.shape[1]),
        }

    fid = _frechet_distance(pack_flow.fused, pack_noflow.fused)
    mmd = _mmd_rbf(pack_flow.fused, pack_noflow.fused, max_points=1000, seed=seed)

    deltas = {
        "pred_rmse_delta_flow_minus_noflow": float(
            analysis_flow["basic"]["pred_rmse"] - analysis_noflow["basic"]["pred_rmse"]
        ),
        "vel_probe_r2_delta_flow_minus_noflow": float(
            analysis_flow["probes"]["velocity_regression"]["r2_mean"]
            - analysis_noflow["probes"]["velocity_regression"]["r2_mean"]
        ),
        "speed_probe_acc_delta_flow_minus_noflow": float(
            analysis_flow["probes"]["speed_bin_probe"]["accuracy"]
            - analysis_noflow["probes"]["speed_bin_probe"]["accuracy"]
        ),
        "direction_probe_acc_delta_flow_minus_noflow": float(
            analysis_flow["probes"]["direction_probe"]["accuracy"]
            - analysis_noflow["probes"]["direction_probe"]["accuracy"]
        ),
        "static_dynamic_probe_acc_delta_flow_minus_noflow": float(
            analysis_flow["probes"]["static_dynamic_probe"]["accuracy"]
            - analysis_noflow["probes"]["static_dynamic_probe"]["accuracy"]
        ),
        "participation_ratio_delta_flow_minus_noflow": float(
            analysis_flow["geometry"]["participation_ratio"]
            - analysis_noflow["geometry"]["participation_ratio"]
        ),
        "intrinsic_dim_delta_flow_minus_noflow": float(
            analysis_flow["geometry"]["intrinsic_dim_lb"]
            - analysis_noflow["geometry"]["intrinsic_dim_lb"]
        ),
        "knn_direction_purity_delta_flow_minus_noflow": float(
            analysis_flow["knn"]["purity_direction_k10"]
            - analysis_noflow["knn"]["purity_direction_k10"]
        ),
    }

    return {
        "layer_similarity": layer_metrics,
        "distribution_shift": {"frechet_distance": float(fid), "mmd_rbf": float(mmd)},
        "deltas": deltas,
    }


def _save_json(path: Path, data: Any) -> None:
    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return "<non-serializable>"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _write_report(
    out_path: Path,
    alignment_stats: dict[str, Any],
    flow_name: str,
    noflow_name: str,
    a_flow: dict[str, Any],
    a_noflow: dict[str, Any],
    cmp: dict[str, Any],
) -> None:
    def fmt(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "N/A"
        return f"{x:.6f}"

    lines = []
    lines.append("# Latent Space Comparison Report")
    lines.append("")
    lines.append(f"- Flow model: `{flow_name}`")
    lines.append(f"- No-flow model: `{noflow_name}`")
    lines.append(f"- Aligned samples: `{alignment_stats['matched']}` (mode: `{alignment_stats['mode']}`)")
    lines.append("")
    lines.append("## Core comparison")
    lines.append("")
    lines.append("| Metric | Flow | No-flow | Delta (Flow - No-flow) |")
    lines.append("|---|---:|---:|---:|")
    rows = [
        ("Pred RMSE", a_flow["basic"]["pred_rmse"], a_noflow["basic"]["pred_rmse"], cmp["deltas"]["pred_rmse_delta_flow_minus_noflow"]),
        ("Velocity probe R2 (mean)", a_flow["probes"]["velocity_regression"]["r2_mean"], a_noflow["probes"]["velocity_regression"]["r2_mean"], cmp["deltas"]["vel_probe_r2_delta_flow_minus_noflow"]),
        ("Speed-bin probe accuracy", a_flow["probes"]["speed_bin_probe"]["accuracy"], a_noflow["probes"]["speed_bin_probe"]["accuracy"], cmp["deltas"]["speed_probe_acc_delta_flow_minus_noflow"]),
        ("Direction probe accuracy", a_flow["probes"]["direction_probe"]["accuracy"], a_noflow["probes"]["direction_probe"]["accuracy"], cmp["deltas"]["direction_probe_acc_delta_flow_minus_noflow"]),
        ("Static/dynamic probe accuracy", a_flow["probes"]["static_dynamic_probe"]["accuracy"], a_noflow["probes"]["static_dynamic_probe"]["accuracy"], cmp["deltas"]["static_dynamic_probe_acc_delta_flow_minus_noflow"]),
        ("Participation ratio", a_flow["geometry"]["participation_ratio"], a_noflow["geometry"]["participation_ratio"], cmp["deltas"]["participation_ratio_delta_flow_minus_noflow"]),
        ("Intrinsic dim (LB)", a_flow["geometry"]["intrinsic_dim_lb"], a_noflow["geometry"]["intrinsic_dim_lb"], cmp["deltas"]["intrinsic_dim_delta_flow_minus_noflow"]),
        ("kNN direction purity", a_flow["knn"]["purity_direction_k10"], a_noflow["knn"]["purity_direction_k10"], cmp["deltas"]["knn_direction_purity_delta_flow_minus_noflow"]),
    ]
    for k, v1, v2, dv in rows:
        lines.append(f"| {k} | {fmt(v1)} | {fmt(v2)} | {fmt(dv)} |")
    lines.append("")
    lines.append("## PCA Variance")
    lines.append("")
    ef = a_flow["geometry"].get("explained_var_top3", [float("nan")] * 3)
    en = a_noflow["geometry"].get("explained_var_top3", [float("nan")] * 3)
    lines.append(f"- Flow explained variance (PC1/PC2/PC3): `{fmt(ef[0])}`, `{fmt(ef[1])}`, `{fmt(ef[2])}`")
    lines.append(f"- No-flow explained variance (PC1/PC2/PC3): `{fmt(en[0])}`, `{fmt(en[1])}`, `{fmt(en[2])}`")
    lines.append("")
    lines.append("## Representation shift")
    lines.append("")
    lines.append("| Layer | CKA | SVCCA |")
    lines.append("|---|---:|---:|")
    for ln, vals in cmp["layer_similarity"].items():
        lines.append(f"| {ln} | {fmt(vals['cka'])} | {fmt(vals['svcca'])} |")
    lines.append("")
    lines.append("## Distribution shift")
    lines.append("")
    lines.append(f"- Fréchet distance: `{fmt(cmp['distribution_shift']['frechet_distance'])}`")
    lines.append(f"- MMD (RBF): `{fmt(cmp['distribution_shift']['mmd_rbf'])}`")
    lines.append("")
    lines.append("## Temporal validity")
    lines.append("")
    lines.append(f"- Flow temporal segments: `{a_flow['temporal']['num_segments']}`")
    lines.append(f"- No-flow temporal segments: `{a_noflow['temporal']['num_segments']}`")
    lines.append("")
    lines.append("## Probe Validity")
    lines.append("")
    for pname in ["speed_bin_probe", "direction_probe", "static_dynamic_probe", "motion_boundary_probe"]:
        deg_f = bool(a_flow["probes"][pname].get("degenerate", False))
        deg_n = bool(a_noflow["probes"][pname].get("degenerate", False))
        lines.append(
            f"- `{pname}`: flow degenerate=`{deg_f}`, no-flow degenerate=`{deg_n}`"
        )

    if len(a_flow.get("sequence_metrics", {})) > 0 and len(a_noflow.get("sequence_metrics", {})) > 0:
        lines.append("")
        lines.append("## Sequence Highlights")
        lines.append("")
        lines.append("| Sequence | Flow RMSE | No-flow RMSE | Delta (Flow - No-flow) |")
        lines.append("|---|---:|---:|---:|")
        common = sorted(set(a_flow["sequence_metrics"]).intersection(a_noflow["sequence_metrics"]))
        rows = []
        for sid in common:
            rf = a_flow["sequence_metrics"][sid]["rmse"]
            rn = a_noflow["sequence_metrics"][sid]["rmse"]
            rows.append((sid, rf, rn, rf - rn))
        rows = sorted(rows, key=lambda x: x[3])[:10]
        for sid, rf, rn, d in rows:
            lines.append(f"| {sid} | {fmt(rf)} | {fmt(rn)} | {fmt(d)} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_pack(
    name: str,
    latents_path: str | None,
    model_dir: str | None,
    model_cfg: str | None,
    dataset_cfg: str | None,
    weights: str | None,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    extracted_npz_path: Path,
) -> LatentPack:
    if latents_path is not None:
        return load_latent_pack(Path(latents_path), name=name)
    if model_dir is not None:
        md = Path(model_dir)
        if model_cfg is None:
            model_cfg = str(md / "model-cfg.yml")
        if dataset_cfg is None:
            dataset_cfg = str(md / "dataset-cfg.yml")
        if weights is None:
            weights = str(md / "best.pth")
    missing = [k for k, v in [("model_cfg", model_cfg), ("dataset_cfg", dataset_cfg), ("weights", weights)] if v is None]
    if missing:
        raise ValueError(f"{name}: when latents are not provided, required args missing: {missing}")
    for label, p in [("model_cfg", model_cfg), ("dataset_cfg", dataset_cfg), ("weights", weights)]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{name}: {label} not found: {p}")
    return extract_latents_from_model(
        name=name,
        model_cfg_path=Path(model_cfg),
        dataset_cfg_path=Path(dataset_cfg),
        weights_path=Path(weights),
        sequences=sequences,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_batches=max_batches,
        save_npz_path=extracted_npz_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare latent spaces between flow-head and no-flow models."
    )
    parser.add_argument("--out", default="plots/latent_compare", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--sequences", default=None, help="Comma-separated sequence ids, e.g. 0004,0010")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap aligned samples for analysis")
    parser.add_argument("--max-corr-dims", type=int, default=128)

    parser.add_argument("--flow-name", default="with_flow")
    parser.add_argument("--noflow-name", default="without_flow")

    parser.add_argument("--flow-latents", default=None, help="Path to flow latent .npz or directory")
    parser.add_argument("--noflow-latents", default=None, help="Path to no-flow latent .npz or directory")
    parser.add_argument("--flow-model-dir", default=None, help="Model folder containing best.pth + model-cfg.yml + dataset-cfg.yml")
    parser.add_argument("--noflow-model-dir", default=None, help="Model folder containing best.pth + model-cfg.yml + dataset-cfg.yml")

    parser.add_argument("--flow-model-cfg", default=None)
    parser.add_argument("--flow-dataset-cfg", default=None)
    parser.add_argument("--flow-weights", default=None)

    parser.add_argument("--noflow-model-cfg", default=None)
    parser.add_argument("--noflow-dataset-cfg", default=None)
    parser.add_argument("--noflow-weights", default=None)

    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available: figures will be skipped, metrics/report will still be generated.")
    else:
        _setup_plot_style()

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = parse_sequences(args.sequences)

    flow_pack = _build_pack(
        name=args.flow_name,
        latents_path=args.flow_latents,
        model_dir=args.flow_model_dir,
        model_cfg=args.flow_model_cfg,
        dataset_cfg=args.flow_dataset_cfg,
        weights=args.flow_weights,
        sequences=sequences,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        extracted_npz_path=out_dir / f"extracted_{args.flow_name}.npz",
    )
    noflow_pack = _build_pack(
        name=args.noflow_name,
        latents_path=args.noflow_latents,
        model_dir=args.noflow_model_dir,
        model_cfg=args.noflow_model_cfg,
        dataset_cfg=args.noflow_dataset_cfg,
        weights=args.noflow_weights,
        sequences=sequences,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        extracted_npz_path=out_dir / f"extracted_{args.noflow_name}.npz",
    )

    flow_pack, noflow_pack, alignment_stats = align_packs(flow_pack, noflow_pack, decimals=4)
    if args.max_samples is not None:
        n = min(len(flow_pack.fused), args.max_samples)
        flow_pack = flow_pack.subset(np.arange(n))
        noflow_pack = noflow_pack.subset(np.arange(n))

    analysis_flow = analyze_pack(flow_pack, seed=args.seed, max_corr_dims=args.max_corr_dims)
    analysis_noflow = analyze_pack(noflow_pack, seed=args.seed + 101, max_corr_dims=args.max_corr_dims)
    comparison = compare_packs(
        pack_flow=flow_pack,
        pack_noflow=noflow_pack,
        analysis_flow=analysis_flow,
        analysis_noflow=analysis_noflow,
        seed=args.seed,
    )

    _plot_pca_compare(flow_pack, noflow_pack, plots_dir / "latent_pca_speed_compare.png", color_mode="speed")
    _plot_pca_compare(flow_pack, noflow_pack, plots_dir / "latent_pca_vz_compare.png", color_mode="vz")
    _plot_pca_3d_compare(flow_pack, noflow_pack, plots_dir / "latent_pca3d_speed_compare.png", color_mode="speed")
    _plot_corr_heatmap(analysis_flow["corr"], plots_dir / f"latent_corr_{args.flow_name}.png", f"{args.flow_name}: target/latent correlation")
    _plot_corr_heatmap(analysis_noflow["corr"], plots_dir / f"latent_corr_{args.noflow_name}.png", f"{args.noflow_name}: target/latent correlation")
    _plot_eigenspectrum(
        analysis_flow["eigs"],
        analysis_noflow["eigs"],
        args.flow_name,
        args.noflow_name,
        plots_dir / "latent_eigenspectrum_compare.png",
    )
    if comparison["layer_similarity"]:
        _plot_layerwise_similarity(comparison["layer_similarity"], plots_dir / "layerwise_similarity.png")
    _plot_probe_scores(
        analysis_flow["probes"],
        analysis_noflow["probes"],
        args.flow_name,
        args.noflow_name,
        plots_dir / "probe_comparison.png",
    )
    _plot_robustness_slices(
        analysis_flow["slice_metrics"],
        analysis_noflow["slice_metrics"],
        args.flow_name,
        args.noflow_name,
        plots_dir / "robustness_slices_rmse.png",
    )

    if flow_pack.attention is not None:
        _plot_attention(flow_pack.attention, plots_dir / f"attention_{args.flow_name}.png", f"{args.flow_name}: mean attention")
    if noflow_pack.attention is not None:
        _plot_attention(noflow_pack.attention, plots_dir / f"attention_{args.noflow_name}.png", f"{args.noflow_name}: mean attention")

    _save_json(out_dir / "alignment_stats.json", alignment_stats)
    _save_json(out_dir / f"analysis_{args.flow_name}.json", analysis_flow)
    _save_json(out_dir / f"analysis_{args.noflow_name}.json", analysis_noflow)
    _save_json(out_dir / "comparison.json", comparison)

    _write_report(
        out_path=out_dir / "report.md",
        alignment_stats=alignment_stats,
        flow_name=args.flow_name,
        noflow_name=args.noflow_name,
        a_flow=analysis_flow,
        a_noflow=analysis_noflow,
        cmp=comparison,
    )

    print(f"Saved analysis to: {out_dir}")
    print(f"Aligned samples: {alignment_stats['matched']} (mode={alignment_stats['mode']})")
    print(f"Flow pred RMSE: {analysis_flow['basic']['pred_rmse']:.6f}")
    print(f"No-flow pred RMSE: {analysis_noflow['basic']['pred_rmse']:.6f}")
    print("Key deltas (flow - noflow):")
    for k, v in comparison["deltas"].items():
        print(f"  - {k}: {v:.6f}")


if __name__ == "__main__":
    main()
