"""Matplotlib-based CH100 upper-computer monitor."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import zmq
from matplotlib.animation import FuncAnimation

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CH100 real-time monitor")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--address", help="Override ZeroMQ connect address")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    address = args.address or config["zmq"]["connect_address"]
    topic = config["zmq"].get("topic", "imu_data")
    max_points = int(config["display"].get("max_points", 500))
    interval = int(config["display"].get("refresh_interval_ms", 40))

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.RCVHWM, 5)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
    subscriber.connect(address)

    fields = {
        "angles": ("roll", "pitch", "yaw"),
        "gyro": ("gyr_x", "gyr_y", "gyr_z"),
        "acc": ("acc_x", "acc_y", "acc_z"),
    }
    buffers = {
        name: deque(maxlen=max_points)
        for group in fields.values()
        for name in group
    }

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    figure.canvas.manager.set_window_title("CH100 Windows Monitor")
    figure.suptitle(f"CH100 real-time data - {address}")

    colors = ("tab:red", "tab:green", "tab:blue")
    lines: dict[str, Any] = {}
    for axis, (group, names) in zip(axes, fields.items()):
        for name, color in zip(names, colors):
            lines[name] = axis.plot([], [], label=name, color=color)[0]
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
        axis.set_ylabel({"angles": "deg", "gyro": "deg/s", "acc": "sensor unit"}[group])
    axes[-1].set_xlabel("samples")

    status = figure.text(0.01, 0.01, "Waiting for CH100 data...", fontsize=9)
    received_frames = 0

    def receive_latest() -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                _, payload = subscriber.recv_multipart(flags=zmq.NOBLOCK)
                latest = json.loads(payload.decode("utf-8"))
            except zmq.Again:
                return latest

    def update(_: int) -> list[Any]:
        nonlocal received_frames
        frame = receive_latest()
        if frame is None:
            return [*lines.values(), status]

        received_frames += 1
        gyr = frame.get("gyr", (0.0, 0.0, 0.0))
        acc = frame.get("acc", (0.0, 0.0, 0.0))
        values = {
            "roll": frame.get("roll", 0.0),
            "pitch": frame.get("pitch", 0.0),
            "yaw": frame.get("yaw", 0.0),
            "gyr_x": gyr[0],
            "gyr_y": gyr[1],
            "gyr_z": gyr[2],
            "acc_x": acc[0],
            "acc_y": acc[1],
            "acc_z": acc[2],
        }
        for name, value in values.items():
            buffers[name].append(float(value))

        sample_count = len(buffers["roll"])
        x_values = range(sample_count)
        for name, line in lines.items():
            line.set_data(x_values, buffers[name])

        for axis in axes:
            axis.set_xlim(0, max(max_points - 1, 1))
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

        status.set_text(
            f"frames={received_frames}  "
            f"roll={values['roll']:+.3f} deg  pitch={values['pitch']:+.3f} deg  "
            f"yaw={values['yaw']:+.3f} deg"
        )
        return [*lines.values(), status]

    animation = FuncAnimation(
        figure,
        update,
        interval=interval,
        blit=False,
        cache_frame_data=False,
    )
    figure._ch100_animation = animation  # Keep a reference for Matplotlib.

    try:
        plt.tight_layout(rect=(0, 0.04, 1, 0.96))
        plt.show()
    finally:
        subscriber.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
