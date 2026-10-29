import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol

BAUDRATE = 460800
SPEEDS = (5.0, 10.0, 15.0)
SAMPLE_INTERVAL = 0.02
SETTLE_BAND = 1.0
SETTLE_TIME = 0.30
MAX_REACH_TIME = 20.0
HOLD_TIME = 1.0
COMMAND_INTERVAL = 0.02
CURRENT_ABORT_THRESHOLD = 40.0


def read_float(drive, name):
    response = drive.read_parameter(name)
    if not response:
        raise RuntimeError(f"读取 {name} 无响应")
    return float(response)


def read_sample(drive):
    position, velocity = (float(value) for value in drive.request_feedback(0).split())
    current = read_float(drive, "axis0.motor.current_control.Iq_measured")
    input_vel = read_float(drive, "axis0.controller.input_vel")
    vbus = read_float(drive, "vbus_voltage")
    return velocity, current, input_vel, vbus


def wait_until_idle(drive):
    drive.motor_velocity(0, 0.0, 0.0)
    deadline = time.monotonic() + 8.0
    stopped_since = None
    while time.monotonic() < deadline:
        _position, velocity = (float(value) for value in drive.request_feedback(0).split())
        state = int(read_float(drive, "axis0.current_state"))
        errors = [
            int(read_float(drive, name))
            for name in ("axis0.error", "axis0.motor.error", "axis0.encoder.error")
        ]
        if errors != [0, 0, 0]:
            raise RuntimeError(f"停止过程中 AXIS0 出现错误：{errors}")
        now = time.monotonic()
        if abs(velocity) <= 0.05:
            stopped_since = stopped_since or now
            if now - stopped_since >= 0.20:
                break
        else:
            stopped_since = None
        time.sleep(0.05)
    else:
        raise RuntimeError(f"等待机械停稳超时，最后反馈速度={velocity:.3f} rev/s")

    drive.write_parameter("axis0.requested_state", 1)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if int(read_float(drive, "axis0.current_state")) == 1:
            return
        time.sleep(0.05)
    raise RuntimeError("无法回到 IDLE")


def run_speed(drive, target):
    drive.write_parameter("axis0.requested_state", 8)
    time.sleep(0.4)
    start = time.monotonic()
    samples = []
    reached_at = None
    stable_since = None
    next_command = start
    while time.monotonic() - start < MAX_REACH_TIME:
        if time.monotonic() >= next_command:
            command_at = time.monotonic()
            drive.motor_velocity(0, target, 0.0)
            next_command = command_at + COMMAND_INTERVAL
        velocity, current, input_vel, vbus = read_sample(drive)
        if abs(current) > CURRENT_ABORT_THRESHOLD:
            raise RuntimeError(
                "Iq measured 超过安全中止阈值："
                f"{current:.3f} A > {CURRENT_ABORT_THRESHOLD:.1f} A"
            )
        now = time.monotonic()
        elapsed = now - start
        samples.append((elapsed, velocity, current, input_vel, vbus, now - command_at))
        if abs(velocity - target) <= SETTLE_BAND:
            stable_since = stable_since or now
            if now - stable_since >= SETTLE_TIME:
                reached_at = elapsed
                break
        else:
            stable_since = None
        time.sleep(max(0.0, SAMPLE_INTERVAL - (time.monotonic() - now)))
    if reached_at is None:
        if target == 15.0 and samples:
            maximum = max(row[1] for row in samples)
            last = samples[-1]
            drive.motor_velocity(0, 0.0, 0.0)
            time.sleep(0.3)
            return {
                "target": target,
                "command": target,
                "reach_time": None,
                "feedback_mean": statistics.mean(row[1] for row in samples[-50:]),
                "feedback_max_error": max(abs(target - row[1]) for row in samples[-50:]),
                "current_peak": max(abs(row[2]) for row in samples),
                "current_mean": statistics.mean(abs(row[2]) for row in samples),
                "max_feedback": maximum,
                "mean_command_latency": statistics.mean(row[5] for row in samples),
                "max_command_latency": max(row[5] for row in samples),
                "last_feedback": last[1],
                "last_input_vel": last[3],
                "last_vbus": last[4],
            }
        maximum = max((row[1] for row in samples), default=0.0)
        last = samples[-1] if samples else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        raise RuntimeError(
            f"{target:g} rev/s 在 {MAX_REACH_TIME:g}s 内未达到；"
            f"最大反馈={maximum:.3f} rev/s，最后反馈={last[1]:.3f} rev/s，"
            f"最后Iq={last[2]:.3f} A"
        )
    hold_start = time.monotonic()
    while time.monotonic() - hold_start < HOLD_TIME:
        if time.monotonic() >= next_command:
            command_at = time.monotonic()
            drive.motor_velocity(0, target, 0.0)
            next_command = command_at + COMMAND_INTERVAL
        velocity, current, input_vel, vbus = read_sample(drive)
        if abs(current) > CURRENT_ABORT_THRESHOLD:
            raise RuntimeError(
                "Iq measured 超过安全中止阈值："
                f"{current:.3f} A > {CURRENT_ABORT_THRESHOLD:.1f} A"
            )
        samples.append((time.monotonic() - start, velocity, current, input_vel, vbus, time.monotonic() - command_at))
        time.sleep(SAMPLE_INTERVAL)
    hold_samples = [row for row in samples if row[0] >= reached_at]
    currents = [abs(row[2]) for row in hold_samples]
    velocities = [row[1] for row in hold_samples]
    return {
        "target": target,
        "command": target,
        "reach_time": reached_at,
        "feedback_mean": statistics.mean(velocities),
        "feedback_max_error": max(abs(target - value) for value in velocities),
        "current_peak": max(currents),
        "current_mean": statistics.mean(currents),
        "max_feedback": max(row[1] for row in samples),
        "mean_command_latency": statistics.mean(row[5] for row in samples),
        "max_command_latency": max(row[5] for row in samples),
        "mean_input_vel": statistics.mean(row[3] for row in hold_samples),
        "mean_vbus": statistics.mean(row[4] for row in hold_samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM33")
    parser.add_argument("--output", default="odrive_speed_test.csv")
    parser.add_argument("--speeds", nargs="+", type=float, default=list(SPEEDS))
    args = parser.parse_args()
    drive = ODriveAsciiProtocol(args.port, BAUDRATE)
    rows = []
    try:
        state = int(read_float(drive, "axis0.current_state"))
        errors = [read_float(drive, name) for name in ("axis0.error", "axis0.motor.error", "axis0.encoder.error")]
        ready = int(read_float(drive, "axis0.encoder.is_ready"))
        if errors != [0.0, 0.0, 0.0] or ready != 1 or state not in (1, 8):
            raise RuntimeError(f"AXIS0 不满足测试条件：state={state}, errors={errors}, encoder_ready={ready}")
        for target in args.speeds:
            print(f"测试 {target:g} rev/s")
            rows.append(run_speed(drive, target))
            wait_until_idle(drive)
            print(rows[-1])
    finally:
        try:
            wait_until_idle(drive)
        finally:
            drive.close()
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("RESULT", " ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, serial.SerialException) as error:
        raise SystemExit(f"测试失败：{error}") from error
