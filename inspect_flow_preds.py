import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image

from elope.datasets import EventProcessor, VariableSequenceLoader, FixedSequenceLoader
from elope.models import build_model
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


# Model run name
MODEL_NAME = "emmnet-angles_20260128_194925"

# Sequence dataset to visualize
DATAPATH = Path("elope_data") / "train"

# Output path
SAVE_ROOT = Path("sequence_flow_preds")

# Frames per sequence and stride
MAX_FRAMES = 200
FRAME_STRIDE = 1

# GIF settings
GIF_DURATION = 3


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
            frames_flow.append(flow_to_rgb(flow_np))

            frame_count += 1

        frames_to_gif(
            out_dir / f"{seq_id}.gif",
            (frames_pos, frames_neg, frames_flow),
            nrows=1,
            ncols=3,
            duration=GIF_DURATION,
            loop=0,
        )

    LOGGER.info(f"Saved flow predictions to: {out_dir}")


if __name__ == "__main__":
    main()
