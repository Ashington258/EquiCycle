"""Read CH100 from a Windows serial port and publish parsed frames via ZeroMQ."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import serial
import zmq

from ch100_protocol import CH100Parser

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CH100 Windows data collector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", help="Override COM port, for example COM22")
    parser.add_argument("--baudrate", type=int, help="Override serial baud rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    serial_config = config["ch100"]
    zmq_config = config["zmq"]

    port = args.port or serial_config["port"]
    baudrate = args.baudrate or serial_config["baudrate"]
    timeout = float(serial_config.get("timeout", 0.1))
    read_size = int(serial_config.get("read_size", 512))
    bind_address = zmq_config["bind_address"]
    topic = zmq_config.get("topic", "imu_data")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = CH100Parser()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 10)
    publisher.bind(bind_address)

    logging.info("Opening CH100 on %s at %d baud", port, baudrate)
    logging.info("Publishing on %s with topic %s", bind_address, topic)

    last_report = time.monotonic()
    latest_frame: dict[str, Any] | None = None

    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as device:
            device.reset_input_buffer()
            while True:
                available = device.in_waiting
                data = device.read(max(1, min(max(available, 1), read_size)))
                for frame in parser.parse(data):
                    latest_frame = frame
                    publisher.send_multipart(
                        [topic.encode("utf-8"), json.dumps(frame).encode("utf-8")]
                    )

                now = time.monotonic()
                if now - last_report >= 1.0:
                    if latest_frame is None:
                        logging.info(
                            "Waiting for valid HI91 frames; crc_errors=%d skipped=%d",
                            parser.crc_errors,
                            parser.skipped_bytes,
                        )
                    else:
                        logging.info(
                            "frames=%d crc_errors=%d roll=%+.3f pitch=%+.3f yaw=%+.3f",
                            parser.valid_frames,
                            parser.crc_errors,
                            latest_frame["roll"],
                            latest_frame["pitch"],
                            latest_frame["yaw"],
                        )
                    last_report = now
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    except serial.SerialException as error:
        logging.error("Serial communication failed: %s", error)
        raise SystemExit(1) from error
    finally:
        publisher.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
