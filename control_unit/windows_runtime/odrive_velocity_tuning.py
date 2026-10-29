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
TARGETS = (10.0, 15.0)
COMMAND_PERIOD = 0.02
SAMPLE_PERIOD = 0.02
RUN_TIME = 4.0
HOLD_TARGET_TIME = 0.5
CURRENT_LIMIT = 50.0
VELOCITY_LIMIT = 35.0
RAMP_RATE = 60.0
CURRENT_ABORT_THRESHOLD = 40.0


def read_float(drive, name):
    return float(drive.read_parameter(name))


def sample(drive):
    sample_start = time.monotonic()
    _position, velocity = (float(value) for value in drive.request_feedback(0).split())
    current = read_float(drive, "axis0.motor.current_control.Iq_measured")
    input_vel = read_float(drive, "axis0.controller.input_vel")
    vbus = read_float(drive, "vbus_voltage")
    state = int(read_float(drive, "axis0.current_state"))
    errors = [
        int(read_float(drive, name))
        for name in ("axis0.error", "axis0.motor.error", "axis0.encoder.error")
    ]
    if state != 8 or errors != [0, 0, 0]:
        raise RuntimeError(f"测试中 AXIS0 状态异常：state={state}, errors={errors}")
    return velocity, current, input_vel, vbus, time.monotonic() - sample_start


def set_parameters(drive, kp, ki):
    drive.write_parameter("axis0.controller.config.vel_gain", kp)
    drive.write_parameter("axis0.controller.config.vel_integrator_gain", ki)
    actual_kp = read_float(drive, "axis0.controller.config.vel_gain")
    actual_ki = read_float(drive, "axis0.controller.config.vel_integrator_gain")
    if abs(actual_kp - kp) > 1e-6 or abs(actual_ki - ki) > 1e-6:
        raise RuntimeError(f"速度环参数写入失败：Kp={actual_kp}, Ki={actual_ki}")


def stop(drive):
    drive.motor_velocity(0, 0.0, 0.0)
    deadline = time.monotonic() + 8.0
    stopped_since = None
    velocity = 0.0
    while time.monotonic() < deadline:
        _position, velocity = (float(value) for value in drive.request_feedback(0).split())
        state = int(read_float(drive, "axis0.current_state"))
        errors = [
            int(read_float(drive, name))
            for name in ("axis0.error", "axis0.motor.error", "axis0.encoder.error")
        ]
        if errors != [0, 0, 0]:
            raise RuntimeError(f"停止过程中 AXIS0 出现错误：errors={errors}")
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
        state = int(read_float(drive, "axis0.current_state"))
        errors = [
            int(read_float(drive, name))
            for name in ("axis0.error", "axis0.motor.error", "axis0.encoder.error")
        ]
        if state == 1:
            if errors != [0, 0, 0]:
                raise RuntimeError(f"AXIS0 出现锁存错误：errors={errors}")
            return
        time.sleep(0.05)
    raise RuntimeError("无法回到 IDLE")


def run_candidate(drive, kp, ki, target, output):
    set_parameters(drive, kp, ki)
    drive.write_parameter("axis0.requested_state", 8)
    time.sleep(0.3)
    start = time.monotonic()
    next_command = start
    rows = []
    command_at = start
    try:
        while time.monotonic() - start < RUN_TIME:
            now = time.monotonic()
            if now >= next_command:
                command_at = now
                drive.motor_velocity(0, target, 0.0)
                next_command = now + COMMAND_PERIOD
            velocity, current, input_vel, vbus, sample_latency = sample(drive)
            if abs(current) > CURRENT_ABORT_THRESHOLD:
                raise RuntimeError(
                    "Iq measured 超过安全中止阈值："
                    f"{current:.3f} A > {CURRENT_ABORT_THRESHOLD:.1f} A"
                )
            rows.append({
                "kp": kp,
                "ki": ki,
                "target": target,
                "elapsed": time.monotonic() - start,
                "command": target,
                "input_vel": input_vel,
                "velocity": velocity,
                "error": velocity - target,
                "current": current,
                "vbus": vbus,
                "query_latency": sample_latency,
                "command_age": time.monotonic() - command_at,
            })
            time.sleep(max(0.0, SAMPLE_PERIOD - (time.monotonic() - now)))
    finally:
        stop(drive)
    steady = [row for row in rows if row["elapsed"] >= RUN_TIME - HOLD_TARGET_TIME]
    peak_abs_error = max(abs(row["error"]) for row in rows if row["elapsed"] >= 0.25)
    steady_abs_error = max(abs(row["error"]) for row in steady)
    summary = {
        "kp": kp,
        "ki": ki,
        "target": target,
        "reach_band_time": next((row["elapsed"] for row in rows if abs(row["error"]) <= 1.0), None),
        "peak_abs_error": peak_abs_error,
        "steady_mean_feedback": statistics.mean(row["velocity"] for row in steady),
        "steady_mean_abs_error": statistics.mean(abs(row["error"]) for row in steady),
        "steady_max_abs_error": steady_abs_error,
        "current_peak": max(abs(row["current"]) for row in rows),
        "current_mean": statistics.mean(abs(row["current"]) for row in steady),
        "vbus_min": min(row["vbus"] for row in rows),
        "query_latency_mean": statistics.mean(row["query_latency"] for row in rows),
        "query_latency_max": max(row["query_latency"] for row in rows),
    }
    print("CANDIDATE", summary)
    with open(output, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM33")
    parser.add_argument("--output", default="odrive_velocity_tuning.csv")
    parser.add_argument("--candidates", nargs="+", default=["0.10,0.08", "0.12,0.08", "0.14,0.08", "0.16,0.08", "0.18,0.06", "0.20,0.04"])
    args = parser.parse_args()
    drive = ODriveAsciiProtocol(args.port, BAUDRATE)
    try:
        if int(read_float(drive, "axis0.current_state")) != 1:
            raise RuntimeError("AXIS0 未处于 IDLE")
        for name, expected in [("axis0.controller.config.vel_limit", VELOCITY_LIMIT), ("axis0.controller.config.vel_ramp_rate", RAMP_RATE), ("axis0.motor.config.current_lim", CURRENT_LIMIT)]:
            actual = read_float(drive, name)
            if abs(actual - expected) > 1e-5:
                raise RuntimeError(f"安全参数不符合预期：{name}={actual}, expected={expected}")
        for item in args.candidates:
            kp, ki = (float(value) for value in item.split(","))
            for target in TARGETS:
                run_candidate(drive, kp, ki, target, args.output)
        set_parameters(drive, 0.10, 0.045)
    finally:
        try:
            stop(drive)
        finally:
            drive.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, serial.SerialException) as error:
        raise SystemExit(f"调参失败：{error}") from error
