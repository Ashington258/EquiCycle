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
    PITCH_STOP_ANGLE,
    control_layer,
    get_pid_parameters,
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
TUNING_LOG_QUEUE_SIZE = 4096
TUNING_LOG_BATCH_SIZE = 50
TUNING_LOG_FLUSH_INTERVAL = 0.5
TUNING_LOG_STOP = object()
TUNING_LOG_FIELDS = (
    "host_time_iso",
    "elapsed_s",
    "pitch_deg",
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
data_queue = queue.Queue()
tuning_log_queue = queue.Queue(maxsize=TUNING_LOG_QUEUE_SIZE)


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
                data_queue.put(frame)
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
        "odrive_input_vel_rev_s": 0.0,
        "odrive_state": 0,
        "odrive_axis_error": 0,
        "odrive_motor_error": 0,
        "odrive_controller_error": 0,
    }
    next_diagnostic_read = 0.0
    try:
        while not stop_event.is_set():
            try:
                if "odrive_current_limit_a" not in diagnostics:
                    diagnostics["odrive_current_limit_a"] = float(
                        odrive_instance.read_parameter("axis0.motor.config.current_lim")
                    )
                feedback = odrive_instance.request_feedback(0)
                motor_current = float(
                    odrive_instance.read_parameter(
                        "axis0.motor.current_control.Iq_measured"
                    )
                )
                iq_setpoint = float(
                    odrive_instance.read_parameter(
                        "axis0.motor.current_control.Iq_setpoint"
                    )
                )
                now = time.monotonic()
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
                    next_diagnostic_read = now + 0.25
                data_queue.put(
                    {
                        "device": "odrive",
                        "feedback": feedback,
                        "motor_current_a": motor_current,
                        "iq_setpoint_a": iq_setpoint,
                        **diagnostics,
                    }
                )
            except Exception:
                logging.exception("ODrive 反馈或限制诊断读取失败")
            stop_event.wait(0.02)
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


def tuning_log_thread_function(log_queue=None):
    if log_queue is None:
        log_queue = tuning_log_queue
    log_path, log_file, tuning_writer = create_tuning_log()
    batch = []
    last_flush = time.monotonic()
    logging.info("调参数据开始抓取: %s", log_path)
    try:
        while True:
            timeout = max(
                0.0,
                TUNING_LOG_FLUSH_INTERVAL - (time.monotonic() - last_flush),
            )
            try:
                row = log_queue.get(timeout=timeout)
            except queue.Empty:
                row = TUNING_LOG_STOP

            if row is TUNING_LOG_STOP:
                if batch:
                    tuning_writer.writerows(batch)
                    batch.clear()
                log_file.flush()
                last_flush = time.monotonic()
                if stop_event.is_set() and log_queue.empty():
                    break
            else:
                batch.append(row)

            if batch and (
                len(batch) >= TUNING_LOG_BATCH_SIZE
                or time.monotonic() - last_flush >= TUNING_LOG_FLUSH_INTERVAL
            ):
                tuning_writer.writerows(batch)
                batch.clear()
                log_file.flush()
                last_flush = time.monotonic()
    except Exception:
        logging.exception("调参日志线程错误")
    finally:
        if batch:
            tuning_writer.writerows(batch)
        log_file.flush()
        log_file.close()
        logging.info("调参数据已保存: %s", log_path)
        logging.info("调参日志线程已停止")


def enqueue_tuning_log(row, log_queue=None):
    if log_queue is None:
        log_queue = tuning_log_queue
    try:
        log_queue.put_nowait(row)
        return True
    except queue.Full:
        return False


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


def control_thread_function(odrive_instance):
    next_status_log = 0.0
    dropped_log_rows = 0
    started_at = time.monotonic()
    try:
        while not stop_event.is_set():
            try:
                data = data_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                target_speed = control_layer(data, odrive_instance)
                if data.get("device") == "odrive" and target_speed is not None:
                    if not enqueue_tuning_log(build_tuning_log_row(data, started_at)):
                        dropped_log_rows += 1
                    feedback_values = data["feedback"].split()
                    actual_speed = float(feedback_values[1])
                    now = time.monotonic()
                    if now >= next_status_log:
                        if dropped_log_rows:
                            logging.warning(
                                "调参日志队列已满，累计丢弃 %d 行；控制未被阻塞",
                                dropped_log_rows,
                            )
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
    finally:
        logging.info("控制线程已停止；调参日志累计丢弃 %d 行", dropped_log_rows)


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
    }


def check_axis0_errors(odrive_instance):
    active_errors = {
        name: value
        for name, value in read_axis0_errors(odrive_instance).items()
        if value
    }
    if active_errors:
        try:
            system_error = read_odrive_int(odrive_instance, "error")
            system_diagnostic = f"ODrive error={system_error} (0x{system_error:X})"
        except Exception as exc:
            system_diagnostic = f"ODrive 顶层 error 读取失败：{exc}"
        raise RuntimeError(f"AXIS0 存在错误：{active_errors}；{system_diagnostic}")


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


def configure_axis0_current_limit(odrive_instance):
    target_limit = float(odrive_config["current_limit_a"])
    if not math.isfinite(target_limit) or target_limit <= 0:
        raise ValueError(f"ODrive 电流限制必须为正有限值，收到 {target_limit!r}")
    odrive_instance.write_parameter("axis0.motor.config.current_lim", target_limit)
    actual_limit = float(
        odrive_instance.read_parameter("axis0.motor.config.current_lim")
    )
    if not math.isfinite(actual_limit) or not math.isclose(
        actual_limit, target_limit, rel_tol=0.0, abs_tol=0.01
    ):
        raise RuntimeError(
            f"ODrive 电流限制写入验证失败：目标={target_limit:.2f} A，"
            f"回读={actual_limit:.2f} A"
        )
    logging.info("ODrive AXIS0 电流限制已设置为 %.2f A", actual_limit)


def prepare_axis0_closed_loop(odrive_instance):
    check_axis0_errors(odrive_instance)
    if not read_odrive_int(odrive_instance, "axis0.motor.is_calibrated"):
        raise RuntimeError("AXIS0 电机尚未校准")
    if not read_odrive_int(odrive_instance, "axis0.encoder.is_ready"):
        raise RuntimeError("AXIS0 编码器尚未就绪")

    odrive_instance.motor_velocity(0, 0, 0)
    configure_axis0_current_limit(odrive_instance)
    odrive_instance.write_parameter(
        "axis0.controller.config.input_mode", 1
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
        self_check("ch100", "odrive", config_path=CONFIG_PATH)
    except Exception:
        logging.exception("启动前串口自检失败")
        return

    if input("准备好后输入 ARM 启动；其他输入退出: ").strip() != "ARM":
        print("已取消启动。")
        return

    odrive_instance = ODriveAsciiProtocol(
        port=odrive_config["port"],
        baudrate=odrive_config["baudrate"],
    )
    try:
        prepare_axis0_closed_loop(odrive_instance)
    except Exception:
        logging.exception("AXIS0 闭环启动失败")
        odrive_instance.close()
        return

    log_thread = threading.Thread(
        target=tuning_log_thread_function,
        name="TuningLogThread",
    )
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

    log_thread.start()
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
        log_thread.join()
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
