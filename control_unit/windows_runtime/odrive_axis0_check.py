import argparse
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


DEFAULT_PORT = "COM31"
DEFAULT_BAUDRATE = 460800


def read_float(drive, name):
    response = drive.read_parameter(name)
    if not response:
        raise RuntimeError(f"读取 {name} 时未收到响应")
    return float(response)


def read_int(drive, name):
    return int(read_float(drive, name))


def check_once(drive):
    values = {
        "vbus": read_float(drive, "vbus_voltage"),
        "state": read_int(drive, "axis0.current_state"),
        "axis_error": read_int(drive, "axis0.error"),
        "motor_error": read_int(drive, "axis0.motor.error"),
        "motor_calibrated": read_int(drive, "axis0.motor.is_calibrated"),
        "encoder_error": read_int(drive, "axis0.encoder.error"),
        "encoder_ready": read_int(drive, "axis0.encoder.is_ready"),
        "control_mode": read_int(
            drive, "axis0.controller.config.control_mode"
        ),
        "input_mode": read_int(drive, "axis0.controller.config.input_mode"),
        "input_vel": read_float(drive, "axis0.controller.input_vel"),
    }
    feedback = drive.request_feedback(0).split()
    if len(feedback) < 2:
        raise RuntimeError(f"反馈格式异常：{feedback}")
    values["position"] = float(feedback[0])
    values["velocity"] = float(feedback[1])

    healthy = (
        values["state"] == 8
        and values["axis_error"] == 0
        and values["motor_error"] == 0
        and values["encoder_error"] == 0
        and values["encoder_ready"] == 1
    )
    status = "READY" if healthy else "NOT READY"
    timestamp = time.strftime("%H:%M:%S")
    print(
        f"[{timestamp}] {status} | vbus={values['vbus']:.2f} V | "
        f"state={values['state']} | axis_err={values['axis_error']} | "
        f"motor_err={values['motor_error']} | "
        f"encoder_err={values['encoder_error']} | "
        f"encoder_ready={values['encoder_ready']}"
    )
    print(
        f"           control_mode={values['control_mode']} | "
        f"input_mode={values['input_mode']} | "
        f"input_vel={values['input_vel']:.3f} rev/s | "
        f"position={values['position']:.3f} turns | "
        f"velocity={values['velocity']:.3f} rev/s"
    )
    return healthy


def main():
    parser = argparse.ArgumentParser(
        description="只读检查 ODrive AXIS0；不会清错、校准或发送运动命令。"
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="每秒重复检查，按 Ctrl+C 停止。",
    )
    args = parser.parse_args()

    try:
        drive = ODriveAsciiProtocol(args.port, args.baudrate)
    except serial.SerialException as error:
        raise SystemExit(
            f"无法打开 {args.port}：{error}\n"
            "请确认端口号正确，并关闭占用该串口的程序。"
        ) from error

    try:
        while True:
            check_once(drive)
            if not args.watch:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n检查已停止。")
    except (ValueError, RuntimeError, serial.SerialException) as error:
        raise SystemExit(f"检查失败：{error}") from error
    finally:
        drive.close()


if __name__ == "__main__":
    main()
