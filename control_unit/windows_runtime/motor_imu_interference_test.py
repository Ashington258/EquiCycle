import argparse
import csv
import json
import math
import sys
import threading
import time
from pathlib import Path

import serial
import zmq

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR / "src"))

from ch100_protocol.ch100_protocol import CH100Device
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


CONFIG_PATH = RUNTIME_DIR / "src" / "config.json"
LOG_DIR = RUNTIME_DIR / "logs"
STAGES = (
    (1, 0.0, 10.0),
    (2, 5.0, 10.0),
    (3, 10.0, 10.0),
    (4, 15.0, 10.0),
    (5, 20.0, 10.0),
    (6, 25.0, 10.0),
    (7, 0.0, 20.0),
)


def read_float(drive, name):
    response = drive.read_parameter(name)
    if not response:
        raise RuntimeError(f"读取 {name} 时未收到响应")
    return float(response)


def check_axis_ready(drive):
    values = {
        "state": int(read_float(drive, "axis0.current_state")),
        "axis_error": int(read_float(drive, "axis0.error")),
        "motor_error": int(read_float(drive, "axis0.motor.error")),
        "encoder_error": int(read_float(drive, "axis0.encoder.error")),
        "encoder_ready": int(read_float(drive, "axis0.encoder.is_ready")),
        "control_mode": int(
            read_float(drive, "axis0.controller.config.control_mode")
        ),
        "input_mode": int(
            read_float(drive, "axis0.controller.config.input_mode")
        ),
    }
    if (
        values["state"] != 8
        or values["axis_error"]
        or values["motor_error"]
        or values["encoder_error"]
        or values["encoder_ready"] != 1
    ):
        raise RuntimeError(f"ODrive AXIS0 未就绪：{values}")
    if values["control_mode"] != 2 or values["input_mode"] not in (1, 2):
        raise RuntimeError(
            "ODrive AXIS0 必须已处于速度控制模式："
            f"control_mode={values['control_mode']}, "
            f"input_mode={values['input_mode']}（要求 control_mode=2，input_mode=1 或 2）"
        )
    return values


def parse_feedback(feedback):
    values = feedback.split()
    if len(values) < 2:
        raise RuntimeError(f"ODrive 反馈格式异常：{feedback!r}")
    position, velocity = map(float, values[:2])
    if not math.isfinite(position) or not math.isfinite(velocity):
        raise RuntimeError(f"ODrive 反馈不是有限数值：{feedback!r}")
    return position, velocity


class TestState:
    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = None
        self.stage = 0
        self.target_speed = 0.0
        self.actual_speed = 0.0

    def start_stage(self, stage, target_speed):
        with self.lock:
            if self.started_at is None:
                self.started_at = time.monotonic()
            self.stage = stage
            self.target_speed = target_speed

    def set_actual_speed(self, actual_speed):
        with self.lock:
            self.actual_speed = actual_speed

    def snapshot(self):
        with self.lock:
            elapsed = (
                0.0
                if self.started_at is None
                else time.monotonic() - self.started_at
            )
            return elapsed, self.stage, self.target_speed, self.actual_speed


def imu_logging_worker(
    device,
    state,
    log_path,
    stop_event,
    first_frame_event,
    errors,
    publisher=None,
    topic=b"imu_data",
):
    fieldnames = (
        "host_time_iso",
        "elapsed_s",
        "stage",
        "target_speed_rev_s",
        "actual_speed_rev_s",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "acc_x",
        "acc_y",
        "acc_z",
        "ch100_system_time_ms",
    )
    try:
        device.open()
        with log_path.open("w", newline="", encoding="utf-8-sig") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()
            while not stop_event.is_set():
                for frame in device.read_and_parse():
                    angles = tuple(
                        float(frame[name]) for name in ("roll", "pitch", "yaw")
                    )
                    gyr = tuple(float(value) for value in frame["gyr"])
                    acc = tuple(float(value) for value in frame["acc"])
                    if len(gyr) != 3 or len(acc) != 3:
                        raise RuntimeError(
                            f"CH100 三轴数据长度异常：gyr={gyr!r}, acc={acc!r}"
                        )
                    values = angles + gyr + acc
                    if not all(math.isfinite(value) for value in values):
                        raise RuntimeError(f"CH100 数据不是有限数值：{values!r}")
                    if publisher is not None:
                        publisher.send_multipart(
                            [topic, json.dumps(frame).encode("utf-8")]
                        )
                    first_frame_event.set()
                    elapsed, stage, target_speed, actual_speed = state.snapshot()
                    writer.writerow(
                        {
                            "host_time_iso": time.strftime(
                                "%Y-%m-%dT%H:%M:%S", time.localtime()
                            ),
                            "elapsed_s": f"{elapsed:.6f}",
                            "stage": stage,
                            "target_speed_rev_s": f"{target_speed:.3f}",
                            "actual_speed_rev_s": f"{actual_speed:.6f}",
                            "roll_deg": f"{angles[0]:.6f}",
                            "pitch_deg": f"{angles[1]:.6f}",
                            "yaw_deg": f"{angles[2]:.6f}",
                            "gyr_x": f"{gyr[0]:.6f}",
                            "gyr_y": f"{gyr[1]:.6f}",
                            "gyr_z": f"{gyr[2]:.6f}",
                            "acc_x": f"{acc[0]:.6f}",
                            "acc_y": f"{acc[1]:.6f}",
                            "acc_z": f"{acc[2]:.6f}",
                            "ch100_system_time_ms": frame.get(
                                "system_time_ms", ""
                            ),
                        }
                    )
                log_file.flush()
    except Exception as error:
        errors.append(error)
        stop_event.set()
    finally:
        device.close()


def run_stages(drive, state, stop_event, errors):
    next_feedback_at = 0.0
    next_status_at = 0.0
    latest_position = 0.0
    latest_speed = 0.0

    for stage, target_speed, duration in STAGES:
        if stop_event.is_set():
            break
        check_axis_ready(drive)
        drive.motor_velocity(0, target_speed, 0)
        state.start_stage(stage, target_speed)
        print(
            f"阶段 {stage}/7：目标 {target_speed:.1f} rev/s，"
            f"持续 {duration:.0f} s"
        )
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not stop_event.is_set():
            now = time.monotonic()
            if now >= next_feedback_at:
                latest_position, latest_speed = parse_feedback(
                    drive.request_feedback(0)
                )
                state.set_actual_speed(latest_speed)
                next_feedback_at = now + 0.2
            if now >= next_status_at:
                print(
                    f"  stage={stage} | target={target_speed:+.1f} rev/s | "
                    f"actual={latest_speed:+.3f} rev/s | "
                    f"position={latest_position:+.3f} turns"
                )
                next_status_at = now + 1.0
            stop_event.wait(min(0.02, max(0.0, deadline - time.monotonic())))
        if errors:
            raise RuntimeError(f"CH100 采集失败：{errors[0]}") from errors[0]


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def main():
    config = load_config()
    parser = argparse.ArgumentParser(
        description="按 0/5/10/15/20/25/0 rev/s 七阶段运行 AXIS0，并记录 CH100 姿态角。"
    )
    parser.add_argument("--ch100-port", default=config["ch100"]["port"])
    parser.add_argument("--odrive-port", default=config["odrive"]["port"])
    parser.add_argument(
        "--baudrate", type=int, default=config["odrive"]["baudrate"]
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / time.strftime("motor_imu_test_%Y%m%d_%H%M%S.csv")
    stop_event = threading.Event()
    first_frame_event = threading.Event()
    errors = []
    state = TestState()
    drive = None
    imu_thread = None
    zmq_context = None
    publisher = None

    print("电机对 CH100 姿态角影响测试（总时长 80 秒）")
    print("警告：最高速度 25 rev/s。请架空并牢固固定机构、清空周围人员、准备急停。")
    print("程序不会自动清错、校准或切换 ODrive 控制模式。")
    if input("确认安全条件满足后输入 RUN TEST：").strip() != "RUN TEST":
        print("已取消，未发送运动命令。")
        return

    try:
        drive = ODriveAsciiProtocol(args.odrive_port, args.baudrate)
        axis = check_axis_ready(drive)
        position, velocity = parse_feedback(drive.request_feedback(0))
        if abs(velocity) > 0.2:
            raise RuntimeError(
                f"启动前 AXIS0 仍在转动（{velocity:.3f} rev/s），已拒绝测试"
            )
        drive.motor_velocity(0, 0, 0)
        print(
            f"ODrive 就绪：state={axis['state']}，control_mode=2，"
            f"input_mode={axis['input_mode']}，position={position:.3f} turns"
        )

        zmq_context = zmq.Context()
        publisher = zmq_context.socket(zmq.PUB)
        publisher.setsockopt(zmq.SNDHWM, 10)
        publisher.bind(f"tcp://*:{config['monitoring']['zmq_port']}")
        topic = config["monitoring"].get("topic", "imu_data").encode("utf-8")
        print(
            "实时曲线数据已发布："
            f"tcp://127.0.0.1:{config['monitoring']['zmq_port']}"
        )

        imu_device = CH100Device(args.ch100_port, config["ch100"]["baudrate"])
        imu_thread = threading.Thread(
            target=imu_logging_worker,
            args=(
                imu_device,
                state,
                log_path,
                stop_event,
                first_frame_event,
                errors,
                publisher,
                topic,
            ),
            name="CH100Logger",
        )
        imu_thread.start()
        if not first_frame_event.wait(3.0):
            detail = f"：{errors[0]}" if errors else ""
            raise RuntimeError(f"3 秒内未收到 CH100 数据{detail}")

        print(f"CH100 数据正常，记录文件：{log_path}")
        run_stages(drive, state, stop_event, errors)
        if not stop_event.is_set():
            print("七阶段测试完成。")
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止电机。")
    except (OSError, ValueError, RuntimeError, serial.SerialException) as error:
        print(f"测试终止：{error}")
    finally:
        stop_event.set()
        if drive is not None:
            try:
                drive.motor_velocity(0, 0, 0)
                print("已向 ODrive AXIS0 发送 0 rev/s。")
            except Exception as error:
                print(f"警告：发送零速度失败，请立即使用急停：{error}")
            finally:
                drive.close()
        if imu_thread is not None:
            imu_thread.join(timeout=2.0)
        if publisher is not None:
            publisher.close(linger=0)
        if zmq_context is not None:
            zmq_context.term()
        if log_path.exists():
            print(f"IMU 完整记录已保存：{log_path}")


if __name__ == "__main__":
    main()
