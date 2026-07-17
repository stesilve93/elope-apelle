import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elope.datasets import EventProcessor, VariableSequenceLoader, FixedSequenceLoader
from elope.models import build_model
from elope.evflow import EVFlowNet
from elope.utils import LOGGER, getfiles, load_yaml, increment_path


def flow_to_rgb(flow: np.ndarray) -> Image.Image:
    """Convert optical flow (H, W, 2) to RGB image (H, W, 3) using HSV mapping."""
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)

    hue = (angle + np.pi) / (2 * np.pi)
    hue = (hue * 179).astype(np.uint8)

    mag_norm = np.clip(magnitude / (np.percentile(magnitude, 99) + 1e-6), 0, 1)
    value = (mag_norm * 255).astype(np.uint8)
    saturation = np.ones_like(value, dtype=np.uint8) * 255

    img_hsv = np.stack((hue, saturation, value), axis=-1)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_rgb)


def overlay_flow_arrows(
    base_img: Image.Image,
    flow: np.ndarray,
    step: int = 12,
    scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1
) -> Image.Image:
    """Overlay sparse flow vectors as arrows on an RGB image."""
    img = np.array(base_img)
    h, w = img.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            u, v = flow[y, x]
            x2 = int(round(x + u * scale))
            y2 = int(round(y + v * scale))
            if x2 == x and y2 == y:
                continue
            cv2.arrowedLine(img, (x, y), (x2, y2), color, thickness, tipLength=0.3)
    return Image.fromarray(img)


def events_to_image(
    events: np.ndarray, polarity: str, bg_white: bool = True
) -> Image.Image:
    """Render event tensor to RGB for a single polarity."""
    assert polarity in ("positive", "negative")
    disp_pos = polarity == "positive"
    ev = events[0] if disp_pos else events[1]
    img = ev.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()

    h, w = img.shape
    if bg_white:
        out = np.ones((h, w, 3), dtype=np.float32)
        out[..., 2 if disp_pos else 0] -= img
        out[..., 1] -= img
    else:
        out = np.zeros((h, w, 3), dtype=np.float32)
        out[..., 0 if disp_pos else 2] = img

    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def frames_to_gif(
    filename: Path,
    frames: list | tuple,
    nrows: int = 1,
    ncols: int = 1,
    loop: int = 0,
    **kwargs
):
    """Save a grid of frames as a GIF."""
    if isinstance(frames[0], Image.Image):
        frames = [frames]

    h, w = frames[0][0].height, frames[0][0].width

    xy = []
    for row in range(nrows):
        for col in range(ncols):
            xy.append((col * w, (nrows - row - 1) * h))

    outframes = []
    for fs in zip(*frames):
        img = Image.new("RGBA", (w * ncols, h * nrows))
        for (k, fk) in enumerate(fs):
            img.paste(fk.convert("RGBA"), xy[k])
        outframes.append(img)

    outframes[0].save(
        filename,
        save_all=True,
        append_images=outframes[1:],
        loop=loop,
        **kwargs
    )


def build_evflownet_tensor(
    full_events,
    time_s: float,
    integration_window_us: float,
    height: int,
    width: int,
    side: str = "left",
) -> np.ndarray:
    """Return event tensor shaped (2, 2, H, W) for EVFlowNet input."""
    if side == "left":
        t_end = 1e6 * time_s
        t_beg = t_end - integration_window_us
    else:
        t_beg = 1e6 * time_s
        t_end = t_beg + integration_window_us

    mask = (full_events["t"] >= t_beg) & (full_events["t"] <= t_end)
    events = full_events[mask].copy()

    if events.dtype == [("x", "<i2"), ("y", "<i2"), ("p", "?"), ("t", "<i8")]:
        events_array = np.column_stack(
            [events["x"], events["y"], events["p"].astype(int), events["t"]]
        )
    else:
        if events.ndim == 1:
            events_array = np.array(
                [[e[0], e[1], int(e[2]), e[3]] for e in events], dtype=np.float32
            )
        else:
            events_array = events

    if events_array.shape[0] == 0:
        return np.zeros((2, 2, height, width), dtype=np.float32)

    tensor = EventProcessor.events_to_tensor(
        events_array,
        1e6 * time_s,
        height,
        width,
        2,
        method="evflownet",
        time_window=integration_window_us,
        side=side,
        clamp=-1,
    )
    # (T, H, W, 2) -> (2, T, H, W)
    tensor = np.transpose(tensor, (3, 0, 1, 2)).astype(np.float32)
    return tensor


# Model run name
MODEL_NAME = "emmnet-angles-of_20260203_135413"

# Sequence dataset to visualize
DATAPATH = Path("elope_data") / "train"

# Output path
SAVE_ROOT = Path("sequence_flow_preds")

# Frames per sequence and stride
MAX_FRAMES = 200
FRAME_STRIDE = 1
ARROW_STEP = 12
ARROW_SCALE = 1.5
ARROW_THICKNESS = 1

# GIF settings
GIF_DURATION = 3
USE_EVFLOWNET = True
EVFLOWNET_WEIGHTS = Path("weights") / "evflownet" / "evflownet.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    model_path = Path("weights") / MODEL_NAME
    model_cfg_path = model_path / "model-cfg.yml"
    dataset_cfg_path = model_path / "dataset-cfg.yml"
    weights_path = model_path / "best.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model_cfg = load_yaml(model_cfg_path)
    dataset_cfg = load_yaml(dataset_cfg_path)

    # Build model and load weights
    model = build_model(model_cfg, dataset_cfg, device=device)
    data = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(data, strict=False)
    model.eval()
    model.to(device)

    evflow = None
    if USE_EVFLOWNET:
        if not EVFLOWNET_WEIGHTS.exists():
            raise FileNotFoundError(f"EVFlowNet weights not found: {EVFLOWNET_WEIGHTS}")
        evflow = EVFlowNet(batch_norm=True)
        data = torch.load(str(EVFLOWNET_WEIGHTS), map_location=device)
        evflow.load_state_dict(data)
        evflow.eval()
        evflow.to(device)

    # Build sequence loader
    if dataset_cfg["sequence_type"] == "fixed":
        seq_cls = FixedSequenceLoader
    else:
        seq_cls = VariableSequenceLoader

    events_cfg = dataset_cfg["events"]
    int_window = float(events_cfg["integration_window"])
    event_integration_window = model_cfg.get("event_integration_window", int_window)
    event_normalization = model_cfg.get("event_normalization", "null")

    seq_loader = seq_cls(
        DATAPATH,
        time_step=float(dataset_cfg.get("time_step", -1)),
        event_integration_window=float(event_integration_window),
        event_encoder_method=events_cfg["encoder_method"],
        event_clamp=int(events_cfg.get("clamp", -1)),
        event_H=int(events_cfg["height"]),
        event_W=int(events_cfg["width"]),
        event_T=int(events_cfg["channels"]),
        sequence_len=int(model_cfg["sequence_length"]),
        sequence_pad=model_cfg["padding"],
    )

    out_dir = increment_path(SAVE_ROOT / MODEL_NAME, exist_ok=False)
    out_dir.mkdir(parents=True)

    seq_files = getfiles(DATAPATH)
    sequences = [s.stem for s in seq_files]
    sequences.sort()

    for seq_id in sequences:
        LOGGER.info(f"Rendering sequence: {seq_id}")
        seq_loader.load_sequence(seq_id, events_side="left", test=False)

        frames_pos = []
        frames_neg = []
        frames_flow = []
        frames_evflow = []

        idx_beg = seq_loader.out_len - 1
        frame_count = 0
        for k in range(idx_beg, len(seq_loader), FRAME_STRIDE):
            if frame_count >= MAX_FRAMES:
                break

            data_k = seq_loader.get_data_at_index(k)
            tms = data_k["times"].unsqueeze(0).to(device)
            imu = data_k["imu"].unsqueeze(0).to(device)
            ranges = data_k["rangemeter"].unsqueeze(0).to(device)
            events = data_k["events"].unsqueeze(0).to(device)

            if event_normalization != "null":
                for i in range(events.shape[0]):
                    event_clamp = seq_loader.event_clamp
                    max_val = event_clamp if event_clamp > 0 else None
                    events[i] = EventProcessor.normalize_tensor(
                        events[i], method=event_normalization, max_val=max_val
                    )

            tms_in = tms - tms[..., 0:1]

            with torch.no_grad():
                outputs = model(tms_in, events, imu, ranges)
                flow = outputs.get("flow_prediction", None)

            if flow is None:
                raise RuntimeError("Model did not return flow_prediction. Is flow_aux enabled?")

            flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()

            events_last = events[0, -1].cpu().numpy()  # (2, C, H, W)
            ev_sum = events_last.sum(axis=1)  # (2, H, W)

            frames_pos.append(events_to_image(ev_sum, "positive"))
            frames_neg.append(events_to_image(ev_sum, "negative"))
            flow_img = flow_to_rgb(flow_np)
            flow_img = overlay_flow_arrows(
                flow_img, flow_np, step=ARROW_STEP, scale=ARROW_SCALE, thickness=ARROW_THICKNESS
            )
            frames_flow.append(flow_img)

            if evflow is not None:
                # Build EVFlowNet input from raw events to match analysis/inspect_events.py
                t_ref = float(tms[0, -1].item())
                tensor_ev = build_evflownet_tensor(
                    seq_loader.full_events,
                    t_ref,
                    float(event_integration_window),
                    int(events_cfg["height"]),
                    int(events_cfg["width"]),
                    side="left",
                )
                count = tensor_ev[:, 0]
                stamp = tensor_ev[:, 1]
                _, h, w = count.shape
                event_img = np.vstack((count, stamp)).astype(np.float32)  # (4, H, W)
                x = torch.from_numpy(event_img).unsqueeze(0).to(device)
                x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    output = evflow(x)
                flow_ev = output["flow3"]
                flow_ev = F.interpolate(flow_ev, size=(h, w), mode="bilinear", align_corners=False)
                flow_ev = flow_ev.squeeze(0).permute(1, 2, 0).cpu().numpy()
                flow_ev = np.flip(flow_ev, 2)

                flow_ev_img = flow_to_rgb(flow_ev)
                flow_ev_img = overlay_flow_arrows(
                    flow_ev_img, flow_ev, step=ARROW_STEP, scale=ARROW_SCALE, thickness=ARROW_THICKNESS
                )
                frames_evflow.append(flow_ev_img)

            frame_count += 1

        frames_to_gif(
            out_dir / f"{seq_id}.gif",
            (frames_pos, frames_neg, frames_flow, frames_evflow) if evflow is not None else (frames_pos, frames_neg, frames_flow),
            nrows=1,
            ncols=4 if evflow is not None else 3,
            duration=GIF_DURATION,
            loop=0,
        )

    LOGGER.info(f"Saved flow predictions to: {out_dir}")


if __name__ == "__main__":
    main()
