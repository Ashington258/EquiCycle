import json
import logging
from pathlib import Path

import serial


def load_device_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.json"
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def self_check(*device_names, config_path=None):
    device_config = load_device_config(config_path)
    for device_name in device_names:
        if device_name not in device_config:
            raise ValueError(f"未知设备名称: {device_name}")

        port = device_config[device_name]["port"]
        baudrate = device_config[device_name]["baudrate"]
        try:
            with serial.Serial(port=port, baudrate=baudrate, timeout=1):
                logging.info(
                    "%s 串口自检通过: %s @ %s",
                    device_name.upper(),
                    port,
                    baudrate,
                )
        except (serial.SerialException, OSError) as error:
            raise RuntimeError(
                f"{device_name.upper()} 串口自检失败 ({port}): {error}"
            ) from error
