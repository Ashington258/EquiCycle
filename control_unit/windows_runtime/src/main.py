import csv
import json
import logging
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import zmq

from balance_control.balance_control import (
    BALANCE_MODE_ACTIVE,
    MOTOR_SPEED_SLEW_RATE,
    PITCH_STOP_ANGLE,
    adjust_motor_speed,
    get_pid_parameters,
    update_control_data,
    get_balance_status,
    get_pitch_status,
    get_tuning_snapshot,
    set_balance_mode,
)
from ch100_protocol.ch100_protocol import CH100Device
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol
from self_check.self_check import self_check


APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = APP_DIR.parent
CONFIG_PATH = APP_DIR / "config.json"
LOG_DIR = RUNTIME_DIR / "logs"
TUNING_LOG_FIELDS = (
    "host_time_iso",
    "elapsed_s",
    "pitch_raw_deg",
    "pitch_deg",
    "gyro_x_raw_deg_s",
    "gyro_x_filtered_deg_s",
    "gyro_x_deg_s",
    "motor_speed_rev_s",
    "velocity_loop_output",
    "angle_loop_output",
    "raw_cascade_output_rev_s",
    "limited_output_rev_s",
    "motor_current_a",
    "iq_setpoint_a",
    "odrive_current_limit_a",
    "odrive_input_vel_rev_s",
    "odrive_state",
    "odrive_axis_error",
    "odrive_motor_error",
    "odrive_controller_error",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


config = load_config()
ch100_config = config["ch100"]
odrive_config = config["odrive"]
monitoring_config = config["monitoring"]
stop_event = threading.Event()
control_data_queue = queue.Queue()
CONTROL_PERIOD = 0.01
ODRIVE_FEEDBACK_PERIOD = 0.01
ODRIVE_TELEMETRY_PERIOD = 0.1
ODRIVE_DIAGNOSTIC_PERIOD = 0.5
ODRIVE_CURRENT_LIMIT_A = 30.0
ODRIVE_TORQUE_LIMIT_NM = 1.0
ODRIVE_VELOCITY_LIMIT_REV_S = 15.0
ODRIVE_VELOCITY_RAMP_RATE_REV_S2 = 60.0

ODRIVE_ERROR_NAMES = {
    "axis0.error": {
        0x00000001: "INVALID_STATE",
        0x00000040: "MOTOR_FAILED",
        0x00000080: "SENSORLESS_ESTIMATOR_FAILED",
        0x00000100: "ENCODER_FAILED",
        0x00000200: "CONTROLLER_FAILED",
        0x00000800: "WATCHDOG_TIMER_EXPIRED",
        0x00004000: "ESTOP_REQUESTED",
        0x00040000: "OVER_TEMP",
    },
    "axis0.motor.error": {
        0x00000001: "PHASE_RESISTANCE_OUT_OF_RANGE",
        0x00000002: "PHASE_INDUCTANCE_OUT_OF_RANGE",
        0x00000008: "DRV_FAULT",
        0x00000010: "CONTROL_DEADLINE_MISSED",
        0x00000080: "MODULATION_MAGNITUDE",
        0x00000400: "CURRENT_SENSE_SATURATION",
        0x00001000: "CURRENT_LIMIT_VIOLATION",
        0x00010000: "MODULATION_IS_NAN",
        0x00020000: "MOTOR_THERMISTOR_OVER_TEMP",
        0x00040000: "FET_THERMISTOR_OVER_TEMP",
        0x00100000: "CURRENT_MEASUREMENT_UNAVAILABLE",
        0x00200000: "CONTROLLER_FAILED",
        0x00400000: "I_BUS_OUT_OF_RANGE",
        0x00800000: "BRAKE_RESISTOR_DISARMED",
        0x01000000: "SYSTEM_LEVEL",
        0x10000000: "UNKNOWN_TORQUE",
        0x20000000: "UNKNOWN_CURRENT_COMMAND",
        0x40000000: "UNKNOWN_CURRENT_MEASUREMENT",
        0x80000000: "UNKNOWN_VBUS_VOLTAGE",
    },
    "axis0.controller.error": {
        0x00000001: "OVERSPEED",
        0x00000002: "INVALID_INPUT_MODE",
        0x00000004: "UNSTABLE_GAIN",
        0x00000020: "INVALID_ESTIMATE",
        0x00000080: "SPINOUT_DETECTED",
    },
    "axis0.encoder.error": {
        0x00000001: "UNSTABLE_GAIN",
        0x00000002: "CPR_POLEPAIRS_MISMATCH",
        0x00000004: "NO_RESPONSE",
        0x00000010: "ILLEGAL_HALL_STATE",
        0x00000040: "ABS_SPI_TIMEOUT",
        0x00000080: "ABS_SPI_COM_FAIL",
    },
}


def create_ch100_publisher():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 10)
    publisher.bind(f"tcp://*:{monitoring_config['zmq_port']}")
    return context, publisher


def ch100_thread_function():
    device = CH100Device(
        port=ch100_config["port"],
        baudrate=ch100_config["baudrate"],
    )
    context = None
    publisher = None
    try:
        context, publisher = create_ch100_publisher()
        device.open()
        logging.info("CH100 已连接: %s", ch100_config["port"])
        logging.info(
            "CH100 实时曲线数据已发布: tcp://127.0.0.1:%d",
            monitoring_config["zmq_port"],
        )
        topic = monitoring_config.get("topic", "imu_data").encode("utf-8")
        while not stop_event.is_set():
            for frame in device.read_and_parse():
                publisher.send_multipart(
                    [topic, json.dumps(frame).encode("utf-8")]
                )
                frame["device"] = "ch100"
                control_data_queue.put(frame)
    except Exception:
        logging.exception("CH100 线程错误")
    finally:
        device.close()
        if publisher is not None:
            publisher.close(linger=0)
        if context is not None:
            context.term()
        logging.info("CH100 线程已停止")


def odrive_thread_function(odrive_instance):
    diagnostics = {
        "motor_current_a": 0.0,
        "iq_setpoint_a": 0.0,
        "odrive_input_vel_rev_s": 0.0,
        "odrive_state": 0,
        "odrive_axis_error": 0,
        "odrive_motor_error": 0,
        "odrive_controller_error": 0,
    }
    next_telemetry_read = 0.0
    next_diagnostic_read = 0.0
    next_feedback_at = time.monotonic()
    read_error_active = False
    suppressed_read_errors = 0
    try:
        while not stop_event.is_set():
            try:
                if "odrive_current_limit_a" not in diagnostics:
                    diagnostics["odrive_current_limit_a"] = float(
                        odrive_instance.read_parameter("axis0.motor.config.current_lim")
                    )
                feedback = odrive_instance.request_feedback(0)
                now = time.monotonic()
                if now >= next_telemetry_read:
                    diagnostics.update(
                        {
                            "motor_current_a": float(
                                odrive_instance.read_parameter(
                                    "axis0.motor.current_control.Iq_measured"
                                )
                            ),
                            "iq_setpoint_a": float(
                                odrive_instance.read_parameter(
                                    "axis0.motor.current_control.Iq_setpoint"
                                )
                            ),
                        }
                    )
                    next_telemetry_read = now + ODRIVE_TELEMETRY_PERIOD
                if now >= next_diagnostic_read:
                    diagnostics.update(
                        {
                            "odrive_input_vel_rev_s": float(
                                odrive_instance.read_parameter(
                                    "axis0.controller.input_vel"
                                )
                            ),
                            "odrive_state": read_odrive_int(
                                odrive_instance, "axis0.current_state"
                            ),
                            "odrive_axis_error": read_odrive_int(
                                odrive_instance, "axis0.error"
                            ),
                            "odrive_motor_error": read_odrive_int(
                                odrive_instance, "axis0.motor.error"
                            ),
                            "odrive_controller_error": read_odrive_int(
                                odrive_instance, "axis0.controller.error"
                            ),
                        }
                    )
                    next_diagnostic_read = now + ODRIVE_DIAGNOSTIC_PERIOD
                control_data_queue.put(
                    {
                        "device": "odrive",
                        "feedback": feedback,
                        **diagnostics,
                    }
                )
                if read_error_active:
                    logging.warning(
                        "ODrive 反馈读取已恢复；本次连续失败 %d 次",
                        suppressed_read_errors + 1,
                    )
                    read_error_active = False
                    suppressed_read_errors = 0
            except Exception:
                if not read_error_active:
                    logging.exception("ODrive 反馈或限制诊断读取失败；等待恢复")
                    read_error_active = True
                else:
                    suppressed_read_errors += 1
            next_feedback_at += ODRIVE_FEEDBACK_PERIOD
            now = time.monotonic()
            if next_feedback_at <= now:
                next_feedback_at = now + ODRIVE_FEEDBACK_PERIOD
            stop_event.wait(max(0.0, next_feedback_at - now))
    finally:
        logging.info("ODrive 反馈线程已停止")


def create_tuning_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / time.strftime("balance_tuning_%Y%m%d_%H%M%S.csv")
    log_file = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(log_file, fieldnames=TUNING_LOG_FIELDS)
    writer.writeheader()
    log_file.flush()
    return path, log_file, writer


def build_tuning_log_row(data, started_at):
    snapshot = get_tuning_snapshot()
    odrive_values = {
        "motor_current_a": float(data["motor_current_a"]),
        "iq_setpoint_a": float(data["iq_setpoint_a"]),
        "odrive_current_limit_a": float(data["odrive_current_limit_a"]),
        "odrive_input_vel_rev_s": float(data["odrive_input_vel_rev_s"]),
        "odrive_state": int(data["odrive_state"]),
        "odrive_axis_error": int(data["odrive_axis_error"]),
        "odrive_motor_error": int(data["odrive_motor_error"]),
        "odrive_controller_error": int(data["odrive_controller_error"]),
    }
    values = (*snapshot.values(), *odrive_values.values())
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"调参数据必须为有限数值，收到 {values!r}")
    return {
        "host_time_iso": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "elapsed_s": time.monotonic() - started_at,
        **snapshot,
        **odrive_values,
    }


def record_tuning_log_row(tuning_writer, log_file, data, started_at):
    try:
        tuning_writer.writerow(build_tuning_log_row(data, started_at))
        log_file.flush()
        return True
    except Exception:
        logging.exception("调参 CSV 写入失败，已停止本次调参记录")
        return False


def control_thread_function(odrive_instance):
    next_status_log = 0.0
    next_control_at = time.monotonic()
    latest_odrive_data = None
    odrive_data_updated = False
    started_at = next_control_at
    log_path, log_file, tuning_writer = create_tuning_log()
    logging.info(
        "调参数据开始抓取: %s；平衡控制频率=%.0f Hz",
        log_path,
        1.0 / CONTROL_PERIOD,
    )
    try:
        while not stop_event.is_set():
            timeout = max(0.0, next_control_at - time.monotonic())
            try:
                data = control_data_queue.get(timeout=timeout)
                update_control_data(data)
                if data.get("device") == "odrive":
                    latest_odrive_data = data
                    odrive_data_updated = True
            except queue.Empty:
                pass
            except Exception:
                logging.exception("控制输入处理失败")

            now = time.monotonic()
            if now < next_control_at:
                continue

            try:
                target_speed = adjust_motor_speed(odrive_instance)
                if (
                    odrive_data_updated
                    and latest_odrive_data is not None
                    and tuning_writer is not None
                    and not record_tuning_log_row(
                        tuning_writer, log_file, latest_odrive_data, started_at
                    )
                ):
                    tuning_writer = None
                odrive_data_updated = False

                if latest_odrive_data is not None and now >= next_status_log:
                    actual_speed = float(latest_odrive_data["feedback"].split()[1])
                    pitch, pitch_is_fresh = get_pitch_status()
                    mode, setpoint = get_balance_status()
                    status = (
                        "stale"
                        if not pitch_is_fresh
                        else (
                            "tilt-stop"
                            if abs(pitch - setpoint) > PITCH_STOP_ANGLE
                            else mode
                        )
                    )
                    logging.info(
                        "pitch=%+.3f deg | setpoint=%+.2f deg | "
                        "target=%+.3f rev/s (%+.1f RPM) | "
                        "actual=%+.3f rev/s (%+.1f RPM) | %s",
                        pitch,
                        setpoint,
                        target_speed,
                        target_speed * 60,
                        actual_speed,
                        actual_speed * 60,
                        status,
                    )
                    next_status_log = now + 1.0
            except Exception:
                logging.exception("控制逻辑处理失败")
                try:
                    odrive_instance.motor_velocity(0, 0, 0)
                except Exception:
                    logging.exception("控制出错后停止 ODrive 轴 0 失败")

            next_control_at += CONTROL_PERIOD
            if next_control_at <= now:
                next_control_at = now + CONTROL_PERIOD
    finally:
        log_file.flush()
        log_file.close()
        logging.info("调参数据已保存: %s", log_path)
        logging.info("控制线程已停止")


def read_odrive_int(odrive_instance, property_name):
    response = odrive_instance.read_parameter(property_name)
    if not response:
        raise RuntimeError(f"读取 {property_name} 时未收到响应")
    return int(float(response))


def read_axis0_errors(odrive_instance):
    return {
        "axis0.error": read_odrive_int(odrive_instance, "axis0.error"),
        "axis0.motor.error": read_odrive_int(
            odrive_instance, "axis0.motor.error"
        ),
        "axis0.encoder.error": read_odrive_int(
            odrive_instance, "axis0.encoder.error"
        ),
        "axis0.controller.error": read_odrive_int(
            odrive_instance, "axis0.controller.error"
        ),
    }


def decode_odrive_error(property_name, value):
    names = []
    known_bits = 0
    for bit, name in ODRIVE_ERROR_NAMES.get(property_name, {}).items():
        known_bits |= bit
        if value & bit:
            names.append(name)
    unknown_bits = value & ~known_bits
    if unknown_bits:
        names.append(f"UNKNOWN_BITS_0x{unknown_bits:X}")
    return "|".join(names) if names else "NONE"


def read_axis0_fault_snapshot(odrive_instance):
    properties = (
        "axis0.current_state",
        "vbus_voltage",
        "axis0.motor.config.current_lim",
        "axis0.motor.config.current_lim_margin",
        "axis0.motor.current_control.Iq_setpoint",
        "axis0.motor.current_control.Iq_measured",
    )
    snapshot = {}
    for property_name in properties:
        try:
            snapshot[property_name] = odrive_instance.read_parameter(property_name)
        except Exception as exc:
            snapshot[property_name] = f"读取失败({type(exc).__name__}: {exc})"
    return snapshot


def check_axis0_errors(odrive_instance):
    active_errors = {
        name: value
        for name, value in read_axis0_errors(odrive_instance).items()
        if value
    }
    if active_errors:
        error_details = ", ".join(
            f"{name}={value} (0x{value:X}: {decode_odrive_error(name, value)})"
            for name, value in active_errors.items()
        )
        try:
            system_error = read_odrive_int(odrive_instance, "error")
            system_diagnostic = f"ODrive error={system_error} (0x{system_error:X})"
        except Exception as exc:
            system_diagnostic = f"ODrive 顶层 error 读取失败：{exc}"
        snapshot = read_axis0_fault_snapshot(odrive_instance)
        snapshot_details = ", ".join(
            f"{name}={value}" for name, value in snapshot.items()
        )
        current_limit_guidance = ""
        if active_errors.get("axis0.motor.error", 0) & 0x00001000:
            current_limit_guidance = (
                " CURRENT_LIMIT_VIOLATION 应检查母线压降、相线/电机参数、"
                "current_lim 与 current_lim_margin，不能仅通过提高电流限制处理。"
            )
        raise RuntimeError(
            f"AXIS0 存在保护故障：{error_details}；{system_diagnostic}；"
            f"故障快照：{snapshot_details}。程序未清错、未进入闭环。"
            f"{current_limit_guidance}"
        )


def clear_axis0_errors_while_idle(odrive_instance):
    state = read_odrive_int(odrive_instance, "axis0.current_state")
    if state != 1:
        raise RuntimeError(f"仅允许在 IDLE 状态清错，current_state={state}")

    latched_errors = {
        name: value
        for name, value in read_axis0_errors(odrive_instance).items()
        if value
    }
    odrive_instance.system_command("sc")
    time.sleep(0.2)
    check_axis0_errors(odrive_instance)
    if latched_errors:
        logging.warning("AXIS0 已在 IDLE 状态清除锁存错误：%s", latched_errors)
    else:
        logging.info("AXIS0 已在 IDLE 状态执行清错确认")


def enter_axis0_idle_and_clear_errors(odrive_instance):
    odrive_instance.write_parameter("axis0.requested_state", 1)
    time.sleep(0.1)
    state = read_odrive_int(odrive_instance, "axis0.current_state")
    if state != 1:
        raise RuntimeError(f"STOP 后 AXIS0 未进入 IDLE，current_state={state}")
    clear_axis0_errors_while_idle(odrive_instance)


def configure_axis0_safety_limits(odrive_instance):
    torque_constant = float(
        odrive_instance.read_parameter("axis0.motor.config.torque_constant")
    )
    if not math.isfinite(torque_constant) or torque_constant <= 0:
        raise RuntimeError(f"ODrive 转矩常数无效：{torque_constant!r}")
    torque_limit = ODRIVE_TORQUE_LIMIT_NM
    limits = (
        ("axis0.motor.config.current_lim", ODRIVE_CURRENT_LIMIT_A, "电流", "A"),
        ("axis0.motor.config.torque_lim", torque_limit, "扭矩", "Nm"),
        (
            "axis0.controller.config.vel_limit",
            ODRIVE_VELOCITY_LIMIT_REV_S,
            "速度",
            "rev/s",
        ),
    )
    for parameter, target, label, unit in limits:
        odrive_instance.write_parameter(parameter, target)
        actual = float(odrive_instance.read_parameter(parameter))
        if not math.isfinite(actual) or not math.isclose(
            actual, target, rel_tol=0.0, abs_tol=0.01
        ):
            raise RuntimeError(
                f"ODrive {label}限制写入验证失败：目标={target:.2f} {unit}，"
                f"回读={actual:.2f} {unit}"
            )
    logging.info(
        "ODrive AXIS0 安全限制已验证：电流=%.1f A，扭矩=%.2f Nm（Kt=%.4f Nm/A），速度=%.1f rev/s",
        ODRIVE_CURRENT_LIMIT_A,
        torque_limit,
        torque_constant,
        ODRIVE_VELOCITY_LIMIT_REV_S,
    )


def prepare_axis0_closed_loop(odrive_instance):
    check_axis0_errors(odrive_instance)
    if not read_odrive_int(odrive_instance, "axis0.motor.is_calibrated"):
        raise RuntimeError("AXIS0 电机尚未校准")
    if not read_odrive_int(odrive_instance, "axis0.encoder.is_ready"):
        raise RuntimeError("AXIS0 编码器尚未就绪")

    odrive_instance.motor_velocity(0, 0, 0)
    configure_axis0_safety_limits(odrive_instance)
    odrive_instance.write_parameter(
        "axis0.controller.config.vel_ramp_rate",
        ODRIVE_VELOCITY_RAMP_RATE_REV_S2,
    )
    ramp_rate = float(
        odrive_instance.read_parameter("axis0.controller.config.vel_ramp_rate")
    )
    if not math.isfinite(ramp_rate) or not math.isclose(
        ramp_rate,
        ODRIVE_VELOCITY_RAMP_RATE_REV_S2,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            "ODrive 速度斜坡写入验证失败："
            f"目标={ODRIVE_VELOCITY_RAMP_RATE_REV_S2:.2f} rev/s²，"
            f"回读={ramp_rate:.2f} rev/s²"
        )
    odrive_instance.write_parameter(
        "axis0.controller.config.input_mode", 2
    )
    odrive_instance.write_parameter(
        "axis0.controller.config.control_mode", 2
    )
    odrive_instance.write_parameter("axis0.requested_state", 8)
    time.sleep(0.3)

    state = read_odrive_int(odrive_instance, "axis0.current_state")
    if state != 8:
        raise RuntimeError(f"AXIS0 未进入闭环，current_state={state}")
    logging.info("AXIS0 已进入速度闭环控制")


def read_axis0_speed(odrive_instance):
    response = odrive_instance.request_feedback(0)
    try:
        position, velocity = response.split()
        speed = float(velocity)
        if not math.isfinite(speed) or not math.isfinite(float(position)):
            raise ValueError("non-finite feedback")
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"STOP 速度反馈无效：{response!r}") from exc
    return speed


def stop_axis0_and_enter_idle(odrive_instance):
    stop_error = None
    try:
        check_axis0_errors(odrive_instance)
        state = read_odrive_int(odrive_instance, "axis0.current_state")
        if state != 8:
            raise RuntimeError(f"STOP 前 AXIS0 已退出闭环，current_state={state}")

        speed = read_axis0_speed(odrive_instance)
        initial_speed = speed
        interval = 0.02
        deceleration = 5.0
        started_at = time.monotonic()
        deadline = started_at + abs(speed) / deceleration + 5.0
        while True:
            check_axis0_errors(odrive_instance)
            state = read_odrive_int(odrive_instance, "axis0.current_state")
            if state != 8:
                raise RuntimeError(f"STOP 期间 AXIS0 已退出闭环，current_state={state}")
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(f"STOP 减速超时，实际速度={speed:.3f} rev/s")
            target = math.copysign(
                max(0.0, abs(initial_speed) - deceleration * (now - started_at)),
                initial_speed,
            )
            odrive_instance.motor_velocity(0, target, 0)
            time.sleep(interval)
            speed = read_axis0_speed(odrive_instance)
            if target == 0 and abs(speed) <= 0.05:
                break
    except Exception as exc:
        stop_error = exc
        try:
            if read_odrive_int(odrive_instance, "axis0.current_state") == 8:
                odrive_instance.motor_velocity(0, 0, 0)
        except Exception:
            logging.exception("STOP 异常后无法确认或清零速度目标")

    try:
        enter_axis0_idle_and_clear_errors(odrive_instance)
    except Exception:
        if stop_error is not None:
            logging.error("STOP 原始异常：%s", stop_error)
        raise
    if stop_error is not None:
        raise stop_error


def log_pid_parameters():
    loop_names = {
        "velocity": "速度外环",
        "angle": "角度中环",
        "gyro": "角速度内环",
    }
    logging.info("当前三串级 PID 参数：")
    for loop, parameters in get_pid_parameters().items():
        logging.info(
            "%s: Kp=%g, Ki=%g, Kd=%g, integral_limit=%g, output_limits=%s",
            loop_names[loop],
            parameters["kp"],
            parameters["ki"],
            parameters["kd"],
            parameters["integral_limit"],
            parameters["output_limits"],
        )


def handle_operator_command(command):
    command = command.strip().upper()
    if command == "BALANCE":
        set_balance_mode(BALANCE_MODE_ACTIVE)
        logging.info("已切换到自平衡模式，pitch 目标=0.00°")
        return False
    if command == "STOP":
        mode, _setpoint = get_balance_status()
        if mode != BALANCE_MODE_ACTIVE:
            logging.warning("尚未进入 BALANCE 模式，STOP 指令未执行")
            return False
        logging.info("收到 STOP 指令，正在安全停止")
        return True
    if command:
        logging.warning("未知命令 %r；仅支持 BALANCE 和 STOP", command)
    return False


def main():
    print("Windows CH100 + ODrive pitch 控制")
    print("启动后将向 ODrive 轴 0 发送真实速度命令。")
    print("请先架空驱动轮、确认急停可用，并检查 ODrive 当前状态和残留目标。")

    log_pid_parameters()
    try:
        self_check("ch100", config_path=CONFIG_PATH)
    except Exception:
        logging.exception("启动前 CH100 串口自检失败")
        return

    if input("准备好后输入 ARM 启动；其他输入退出: ").strip() != "ARM":
        print("已取消启动。")
        return

    try:
        odrive_instance = ODriveAsciiProtocol(
            port=odrive_config["port"],
            baudrate=odrive_config["baudrate"],
        )
    except Exception:
        logging.exception("ODrive 串口打开失败")
        return

    try:
        prepare_axis0_closed_loop(odrive_instance)
    except Exception:
        logging.exception("ODrive 已连接，但 AXIS0 闭环启动被拒绝")
        odrive_instance.close()
        return

    worker_threads = [
        threading.Thread(target=ch100_thread_function, name="CH100Thread"),
        threading.Thread(
            target=odrive_thread_function,
            args=(odrive_instance,),
            name="ODriveFeedbackThread",
        ),
        threading.Thread(
            target=control_thread_function,
            args=(odrive_instance,),
            name="ControlThread",
        ),
    ]

    for thread in worker_threads:
        thread.start()
    logging.info(
        "运行中：CH100=%s，ODrive=%s；进入 BALANCE 后输入 STOP 安全停止",
        ch100_config["port"],
        odrive_config["port"],
    )

    try:
        while any(thread.is_alive() for thread in worker_threads):
            command = input(
                "输入 BALANCE 切换到 pitch=0°；进入平衡后输入 STOP 停止: "
            )
            if handle_operator_command(command):
                break
    except KeyboardInterrupt:
        logging.warning("检测到 Ctrl+C，将执行与 STOP 相同的紧急安全停止")
    finally:
        stop_event.set()
        for thread in worker_threads:
            thread.join()
        try:
            stop_axis0_and_enter_idle(odrive_instance)
            logging.info("已将轴 0 减速至零并切换到 IDLE")
        except Exception:
            logging.exception("停止后将 AXIS0 切换到 IDLE 失败")
        finally:
            odrive_instance.close()
        logging.info("已关闭 ODrive 串口")


if __name__ == "__main__":
    main()
