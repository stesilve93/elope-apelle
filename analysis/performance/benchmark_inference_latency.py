"""Benchmark batch-1 inference latency for a saved EMMNet model.

The script loads `model-cfg.yml`, `dataset-cfg.yml`, and `best.pth` from the
selected model directory, creates synthetic inputs with the configured shapes,
and times warmed-up inference on each requested CPU/CUDA device.

Output is written only to stdout: a Markdown table plus a LaTeX-ready summary
sentence. No benchmark files or model artifacts are created or changed.

Run from the repository root, for example:

python analysis/performance/benchmark_inference_latency.py \
  --model-dir weights/emmnet-angles-of_20260209_144255 \
  --devices cpu cuda
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elope.models import build_model  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_state(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def make_inputs(model_cfg: dict, dataset_cfg: dict, device: torch.device) -> tuple:
    sequence_length = int(model_cfg["sequence_length"])
    events_cfg = dataset_cfg["events"]
    time_step = float(dataset_cfg.get("time_step", 0.3))

    events = torch.zeros(
        (
            1,
            sequence_length,
            2,
            int(events_cfg["channels"]),
            int(events_cfg["height"]),
            int(events_cfg["width"]),
        ),
        dtype=torch.float32,
        device=device,
    )
    imu = torch.zeros((1, sequence_length, 6), dtype=torch.float32, device=device)
    rangemeter = torch.ones((1, sequence_length, 1), dtype=torch.float32, device=device)
    times = (
        torch.arange(sequence_length, dtype=torch.float32, device=device).view(1, -1)
        * time_step
    )
    return times, events, imu, rangemeter


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gib = props.total_memory / (1024**3)
        return f"{props.name} ({total_gib:.1f} GiB)"
    cpu_name = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    return f"{cpu_name} ({torch.get_num_threads()} torch threads)"


def benchmark(
    model_dir: Path,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict:
    model_cfg = load_yaml(model_dir / "model-cfg.yml")
    dataset_cfg = load_yaml(model_dir / "dataset-cfg.yml")

    model = build_model(model_cfg, dataset_cfg, device=device)
    model.load_state_dict(load_state(model_dir / "best.pth", device), strict=False)
    model.eval()

    inputs = make_inputs(model_cfg, dataset_cfg, device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(*inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                _ = model(*inputs)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = model(*inputs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "device": str(device),
        "hardware": device_label(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda or "n/a",
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms": elapsed_ms / iterations,
    }


def parse_devices(values: list[str]) -> list[torch.device]:
    devices = []
    for value in values:
        value = value.lower()
        if value == "cpu":
            devices.append(torch.device("cpu"))
        elif value == "cuda":
            if torch.cuda.is_available():
                devices.append(torch.device("cuda:0"))
            else:
                print("Skipping CUDA: torch.cuda.is_available() is False.")
        elif value.startswith("cuda:"):
            index = int(value.split(":", 1)[1])
            if torch.cuda.is_available() and index < torch.cuda.device_count():
                devices.append(torch.device(value))
            else:
                print(f"Skipping {value}: CUDA device is not visible.")
        else:
            raise ValueError(f"Unsupported device: {value}")
    return devices


def print_results(results: list[dict]) -> None:
    print("\nMarkdown:")
    print("| device | hardware | torch | CUDA runtime | latency ms | iterations |")
    print("|:--|:--|:--|:--|--:|--:|")
    for row in results:
        print(
            f"| {row['device']} | {row['hardware']} | {row['torch']} | "
            f"{row['cuda_runtime']} | {row['latency_ms']:.3f} | {row['iterations']} |"
        )

    if results:
        parts = [
            f"{row['device']}: {row['latency_ms']:.3f} ms ({row['hardware']})"
            for row in results
        ]
        print("\nLaTeX-ready sentence:")
        print(
            "Batch-1 inference latency was "
            + "; ".join(parts)
            + f", averaged over {results[0]['iterations']} warm-started forward passes."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("weights/emmnet-angles-of_20260209_144255"),
    )
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    devices = parse_devices(args.devices)
    if not devices:
        raise SystemExit("No requested benchmark devices are visible.")

    results = [
        benchmark(args.model_dir, device, args.warmup, args.iterations)
        for device in devices
    ]
    print_results(results)


if __name__ == "__main__":
    main()
