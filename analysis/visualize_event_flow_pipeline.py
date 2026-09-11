#!/usr/bin/env python3
"""Create a publication-quality visualization of event tensorization and model flow.

The figure shows one model input/output pair:

1. raw events from the causal integration window,
2. tensorized positive/negative event surfaces,
3. the event tensor seen by the model,
4. predicted optical flow as HSV color,
5. sparse flow vectors overlaid on the event tensor,
6. velocity/context annotations.

The script loads a saved ELOPE model and one sample from the selected dataset
split/sequence. It writes a single PNG to `--out` (creating parent directories)
and does not alter the checkpoint or dataset.

Example
-------
python analysis/visualize_event_flow_pipeline.py \\
  --model-dir weights/emmnet-angles-of_20260211_191017 \\
  --split train \\
  --sequence 0010 \\
  --index 72 \\
  --out analysis/plots/figures_paper/event_flow_pipeline_0010.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elope.datasets import EventProcessor, FixedSequenceLoader, VariableSequenceLoader
from elope.models import build_model
from elope.utils import load_yaml


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Map flow angle to hue and robustly normalized magnitude to value."""
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hue = (ang + np.pi) / (2 * np.pi)
    val = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)
    sat = np.ones_like(val) * 0.92
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = colors.hsv_to_rgb(hsv)
    return np.clip(rgb, 0, 1)


def flow_color_wheel(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    radius = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.clip(radius, 0, 1)
    val = np.ones_like(radius)
    wheel = colors.hsv_to_rgb(np.stack([hue, sat, val], axis=-1))
    wheel[radius > 1] = 1.0
    return wheel


def robust_norm(x: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = np.nanpercentile(np.abs(x), percentile)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return np.clip(x / scale, 0, 1)


def event_composite(event_tensor: np.ndarray, dark: bool = True) -> np.ndarray:
    """Render a (2, C, H, W) event tensor as positive/negative RGB composite."""
    ev = np.asarray(event_tensor)
    pos = robust_norm(ev[0].sum(axis=0))
    neg = robust_norm(ev[1].sum(axis=0))
    h, w = pos.shape
    if dark:
        rgb = np.zeros((h, w, 3), dtype=float)
        rgb[..., 0] = 0.06
        rgb[..., 1] = 0.07
        rgb[..., 2] = 0.09
        rgb[..., 0] += 0.95 * pos
        rgb[..., 1] += 0.78 * neg
        rgb[..., 2] += 0.95 * neg
        both = np.minimum(pos, neg)
        rgb += both[..., None] * 0.35
    else:
        rgb = np.ones((h, w, 3), dtype=float)
        rgb[..., 1] -= 0.75 * (pos + neg)
        rgb[..., 2] -= 0.80 * pos
        rgb[..., 0] -= 0.80 * neg
    return np.clip(rgb, 0, 1)


def raw_event_window(full_events: np.ndarray, t_ref_s: float, window_us: float, side: str = "left") -> np.ndarray:
    if side == "left":
        t_end = 1e6 * t_ref_s
        t_beg = t_end - window_us
    else:
        t_beg = 1e6 * t_ref_s
        t_end = t_beg + window_us
    mask = (full_events["t"] >= t_beg) & (full_events["t"] <= t_end)
    return full_events[mask].copy()


def structured_to_columns(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(events) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if events.dtype.names is not None:
        return events["x"], events["y"], events["p"].astype(bool), events["t"].astype(float)
    return events[:, 0], events[:, 1], events[:, 2].astype(bool), events[:, 3].astype(float)


def build_loader(model_cfg: dict, dataset_cfg: dict, split: str):
    events_cfg = dataset_cfg["events"]
    datapath = Path(dataset_cfg["datapath"]).parent / split
    seq_cls = FixedSequenceLoader if dataset_cfg["sequence_type"] == "fixed" else VariableSequenceLoader
    base_window = float(events_cfg["integration_window"])
    event_window = float(model_cfg.get("event_integration_window", base_window))
    return seq_cls(
        datapath,
        time_step=float(dataset_cfg.get("time_step", -1)),
        event_integration_window=event_window,
        event_encoder_method=events_cfg["encoder_method"],
        event_clamp=int(events_cfg.get("clamp", -1)),
        event_H=int(events_cfg["height"]),
        event_W=int(events_cfg["width"]),
        event_T=int(events_cfg["channels"]),
        sequence_len=int(model_cfg["sequence_length"]),
        sequence_pad=model_cfg["padding"],
    )


def choose_index(seq_loader, requested_index: int | None, requested_time: float | None) -> int:
    if requested_index is not None:
        return int(np.clip(requested_index, seq_loader.out_len - 1, len(seq_loader) - 1))
    if requested_time is not None:
        return int(np.argmin(np.abs(seq_loader.seq_times - requested_time)))

    # Default: pick a visually interesting sample with high recent event density,
    # away from the padded beginning and from the final edge.
    densities = []
    start = seq_loader.out_len - 1
    stop = len(seq_loader)
    stride = max(1, (stop - start) // 120)
    for idx in range(start, stop, stride):
        t_ref = float(seq_loader.seq_times[idx])
        events = raw_event_window(seq_loader.full_events, t_ref, seq_loader.event_integration_window)
        densities.append((len(events), idx))
    if not densities:
        return start
    densities.sort(reverse=True)
    return densities[min(4, len(densities) - 1)][1]


def normalize_events_for_model(events: torch.Tensor, seq_loader, method: str) -> torch.Tensor:
    if method == "null":
        return events
    events = events.clone()
    for i in range(events.shape[0]):
        event_clamp = seq_loader.event_clamp
        max_val = event_clamp if event_clamp > 0 else None
        events[i] = EventProcessor.normalize_tensor(events[i], method=method, max_val=max_val)
    return events


def draw_flow_quiver(ax, flow: np.ndarray, step: int, scale: float, color: str = "white") -> None:
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    u = flow[yy, xx, 0]
    v = flow[yy, xx, 1]
    mag = np.sqrt(u * u + v * v)
    keep = mag > np.percentile(mag, 55)
    ax.quiver(
        xx[keep],
        yy[keep],
        u[keep],
        v[keep],
        angles="xy",
        scale_units="xy",
        scale=1.0 / max(scale, 1e-6),
        width=0.0045,
        headwidth=3.2,
        headlength=4.4,
        headaxislength=3.6,
        color=color,
        alpha=0.88,
    )


def draw_raw_events(ax, events: np.ndarray, width: int, height: int, t_ref_us: float) -> None:
    x, y, p, t = structured_to_columns(events)
    ax.set_facecolor("#0b1020")
    if len(x) > 0:
        age = np.clip((t - t.min()) / max(t.max() - t.min(), 1.0), 0, 1)
        pos = p.astype(bool)
        ax.scatter(x[~pos], y[~pos], c=age[~pos], cmap="winter", s=1.4, alpha=0.55, linewidths=0)
        ax.scatter(x[pos], y[pos], c=age[pos], cmap="autumn", s=1.4, alpha=0.62, linewidths=0)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Raw events in window ({len(x):,} events)")
    ax.text(
        0.02,
        0.04,
        "blue/cyan: negative\nred/yellow: positive\nbrighter: newer",
        transform=ax.transAxes,
        color="white",
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", pad=4),
    )


def draw_timeline(ax, raw_events: np.ndarray, t_ref_s: float, window_us: float) -> None:
    _, _, p, t = structured_to_columns(raw_events)
    if len(t) == 0:
        ax.axis("off")
        return
    rel_ms = (t - 1e6 * t_ref_s) / 1000.0
    bins = np.linspace(-window_us / 1000.0, 0.0, 40)
    pos_hist, edges = np.histogram(rel_ms[p], bins=bins)
    neg_hist, _ = np.histogram(rel_ms[~p], bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(centers, pos_hist, color="#ff4d4d", alpha=0.75, label="+")
    ax.fill_between(centers, -neg_hist, color="#38bdf8", alpha=0.75, label="-")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Time before prediction (ms)")
    ax.set_ylabel("Event count")
    ax.set_title("Temporal accumulation")
    ax.legend(loc="upper left", fontsize=8, ncol=2)


def draw_context_box(ax, seq_id: str, idx: int, t_ref: float, pred: np.ndarray, target: np.ndarray | None, flow: np.ndarray) -> None:
    ax.axis("off")
    mag = np.sqrt(np.sum(flow * flow, axis=-1))
    lines = [
        "Model output",
        f"sequence: {seq_id}",
        f"index/time: {idx} / {t_ref:.2f} s",
        "",
        "predicted velocity [m/s]",
        f"vx {pred[0]: .2f}",
        f"vy {pred[1]: .2f}",
        f"vz {pred[2]: .2f}",
        "",
        f"flow median |u|: {np.median(mag):.2f} px",
        f"flow p95 |u|: {np.percentile(mag, 95):.2f} px",
    ]
    if target is not None:
        err = float(np.linalg.norm(pred - target))
        lines += ["", "ground truth velocity [m/s]", f"vx {target[0]: .2f}", f"vy {target[1]: .2f}", f"vz {target[2]: .2f}", f"error norm: {err:.2f}"]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        color="#111827",
        bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", boxstyle="round,pad=0.55"),
    )


def create_figure(
    raw_events: np.ndarray,
    event_tensor: np.ndarray,
    flow: np.ndarray,
    seq_id: str,
    idx: int,
    t_ref: float,
    pred: np.ndarray,
    target: np.ndarray | None,
    window_us: float,
    out_path: Path,
    arrow_step: int,
    arrow_scale: float,
) -> None:
    height, width = event_tensor.shape[-2:]
    composite = event_composite(event_tensor, dark=True)
    flow_rgb = flow_to_rgb(flow)
    pos = event_tensor[0].sum(axis=0)
    neg = event_tensor[1].sum(axis=0)

    fig = plt.figure(figsize=(8.8, 7.2), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 0.42], width_ratios=[1, 1, 1])

    ax_raw = fig.add_subplot(gs[0, 0])
    draw_raw_events(ax_raw, raw_events, width, height, 1e6 * t_ref)

    ax_comp = fig.add_subplot(gs[0, 1])
    ax_comp.imshow(composite)
    ax_comp.set_title("Tensorized event surface")
    ax_comp.set_xticks([])
    ax_comp.set_yticks([])
    ax_comp.text(0.02, 0.04, "red: positive\ncyan: negative", transform=ax_comp.transAxes, color="white", fontsize=8, bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", pad=4))

    ax_pos = fig.add_subplot(gs[0, 2])
    im_pos = ax_pos.imshow(pos, cmap="magma")
    ax_pos.set_title("Positive polarity channel")
    ax_pos.set_xticks([])
    ax_pos.set_yticks([])
    fig.colorbar(im_pos, ax=ax_pos, fraction=0.046, pad=0.02)

    # ax_ctx = fig.add_subplot(gs[:, 3])
    # draw_context_box(ax_ctx, seq_id, idx, t_ref, pred, target, flow)

    ax_neg = fig.add_subplot(gs[1, 0])
    im_neg = ax_neg.imshow(neg, cmap="PuBuGn")
    ax_neg.set_title("Negative polarity channel")
    ax_neg.set_xticks([])
    ax_neg.set_yticks([])
    fig.colorbar(im_neg, ax=ax_neg, fraction=0.046, pad=0.02)

    ax_flow = fig.add_subplot(gs[1, 1])
    ax_flow.imshow(flow_rgb)
    ax_flow.set_title("Predicted optical flow (HSV)")
    ax_flow.set_xticks([])
    ax_flow.set_yticks([])
    wheel_ax = ax_flow.inset_axes([0.69, 0.05, 0.25, 0.25])
    wheel_ax.imshow(flow_color_wheel())
    wheel_ax.set_xticks([])
    wheel_ax.set_yticks([])
    for spine in wheel_ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)
    ax_flow.text(
        0.04,
        0.05,
        "hue: direction\nvalue: speed",
        transform=ax_flow.transAxes,
        color="white",
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", pad=4),
    )

    ax_overlay = fig.add_subplot(gs[1, 2])
    ax_overlay.imshow(composite)
    draw_flow_quiver(ax_overlay, flow, step=arrow_step, scale=arrow_scale, color="yellow")
    ax_overlay.set_title("Flow vectors over event tensor")
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])

    ax_time = fig.add_subplot(gs[2, 0:3])
    draw_timeline(ax_time, raw_events, t_ref, window_us)

    title = "Event tensorization and learned optical-flow head"
    subtitle = f"{seq_id} @ {t_ref:.2f}s | causal window {window_us/1000:.0f} ms | model input -> flow output"
    fig.suptitle(title + "\n" + subtitle, y=0.91, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ELOPE event tensorization and model optical flow.")
    parser.add_argument("--model-dir", type=Path, default=Path("weights/emmnet-angles-of_20260211_191017"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--sequence", default="0010")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--time", type=float, default=None, help="Reference time in seconds. Ignored if --index is set.")
    parser.add_argument("--out", type=Path, default=Path("analysis/plots/figures_paper/event_flow_pipeline.png"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--arrow-step", type=int, default=14)
    parser.add_argument("--arrow-scale", type=float, default=1.4)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg_path = args.model_dir / "model-cfg.yml"
    dataset_cfg_path = args.model_dir / "dataset-cfg.yml"
    weights_path = args.model_dir / "best.pth"
    if not model_cfg_path.exists() or not dataset_cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected model-cfg.yml, dataset-cfg.yml, and best.pth under {args.model_dir}"
        )

    model_cfg = load_yaml(model_cfg_path)
    dataset_cfg = load_yaml(dataset_cfg_path)
    if not bool(model_cfg.get("flow_aux", False)):
        raise ValueError(f"Model config at {model_cfg_path} does not enable flow_aux.")

    model = build_model(model_cfg, dataset_cfg, device=device)
    try:
        state = torch.load(str(weights_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    seq_loader = build_loader(model_cfg, dataset_cfg, args.split)
    seq_loader.load_sequence(args.sequence, events_side="left", test=args.split == "test")
    idx = choose_index(seq_loader, args.index, args.time)
    data = seq_loader.get_data_at_index(idx)

    times = data["times"].unsqueeze(0).to(device)
    imu = data["imu"].unsqueeze(0).to(device)
    ranges = data["rangemeter"].unsqueeze(0).to(device)
    events = data["events"].unsqueeze(0).to(device)
    events = normalize_events_for_model(events, seq_loader, model_cfg.get("event_normalization", "null"))
    tms_in = times - times[..., 0:1]

    with torch.no_grad():
        outputs = model(tms_in, events, imu, ranges)
    flow = outputs.get("flow_prediction")
    if flow is None:
        raise RuntimeError("Model did not return `flow_prediction`. Check that flow_aux is enabled.")

    pred = outputs["prediction"].detach().cpu().numpy().reshape(-1)[:3]
    target = None
    if args.split == "train":
        target = data["states"][-1, 3:6].cpu().numpy().reshape(-1)
    flow_np = flow.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    event_last = events[0, -1].detach().cpu().numpy()
    t_ref = float(data["times"][-1].item())
    raw_events = raw_event_window(seq_loader.full_events, t_ref, seq_loader.event_integration_window, side="left")

    create_figure(
        raw_events=raw_events,
        event_tensor=event_last,
        flow=flow_np,
        seq_id=args.sequence,
        idx=idx,
        t_ref=t_ref,
        pred=pred,
        target=target,
        window_us=float(seq_loader.event_integration_window),
        out_path=args.out,
        arrow_step=args.arrow_step,
        arrow_scale=args.arrow_scale,
    )
    print(f"Saved event-flow pipeline figure to: {args.out}")


if __name__ == "__main__":
    main()
