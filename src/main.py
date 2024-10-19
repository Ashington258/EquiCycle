import threading
from logger.logger import HandleLog
from ch100_protocol.hipnuc_serial_parser_zmq import hipnuc_parser
from odrive_protocol.odrive_protocol_zmq import ODriveAsciiProtocol
from self_check.self_check import self_check
import serial
import zmq

# import logging
import time
import sys
import json

# 配置设备端口号以及波特率


def load_config(file_path="src/config.json"):
    with open(file_path, "r") as f:
        return json.load(f)


config = load_config()

# 使用配置
ch100_port = config["ch100"]["port"]
ch100_baudrate = config["ch100"]["baudrate"]
odrive_port = config["odrive"]["port"]
odrive_baudrate = config["odrive"]["baudrate"]

# # 配置日志
# log.basicConfig(
#     level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )

log = HandleLog()

# 创建一个全局事件
stop_event = threading.Event()


def run_ch100_process(port=ch100_port, baudrate=ch100_baudrate):
    log.info("CH100 线程启动")
    decoder = hipnuc_parser()
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)

    retry_count = 0
    while retry_count < 5:  # 限制重试次数
        try:
            log.info(f"尝试绑定到端口 tcp://*:5555，尝试次数: {retry_count + 1}")
            publisher.bind("tcp://*:5555")
            log.info("端口绑定成功: tcp://*:5555")
            break
        except zmq.error.ZMQError as e:
            log.error(f"绑定失败: {e}，重试中...")
            retry_count += 1
            time.sleep(0.5)  # 缩短重试间隔
    if retry_count == 5:
        log.error("无法绑定端口，CH100进程无法启动")
        raise RuntimeError("无法绑定端口，CH100进程无法启动")

    log.info("开始 CH100 数据采集和发布...")

    try:
        while not stop_event.is_set():  # 检查事件
            data = ser.read(1024)
            if data:
                frames = decoder.parse(data)
                for frame in frames:
                    publisher.send_json(frame)
                    log.info(f"已发布帧: {frame}")
    except Exception as e:
        log.error(f"发生错误: {e}")
    finally:
        ser.close()
        log.info("串口已关闭")
        publisher.close()
        log.info("ZMQ发布者已关闭")
        context.term()
        time.sleep(1)  # 确保有足够时间释放端口
        log.info("ZMQ上下文已终止，CH100 进程关闭")


def run_odrive_process(port=odrive_port, baudrate=odrive_baudrate):
    odrive = ODriveAsciiProtocol(port=port, baudrate=baudrate)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)

    while True:
        try:
            log.info("尝试绑定到端口 tcp://*:5556")
            publisher.bind("tcp://*:5556")
            log.info("端口绑定成功: tcp://*:5556")
            break
        except zmq.error.ZMQError as e:
            log.error(f"绑定失败: {e}，重试中...")
            time.sleep(1)

    while not stop_event.is_set():  # 检查事件
        try:
            odrive.motor_velocity(0, 8)
            odrive.request_feedback(0)
            feedback = odrive.request_feedback(0)
            publisher.send_string(f"motor_speed {feedback}")
        except Exception as e:
            log.error(f"ODrive 进程发生错误: {e}")
            break
    odrive.close()
    log.info("ODrive 串口已关闭")
    publisher.close()
    log.info("ZMQ发布者已关闭")
    context.term()
    time.sleep(1)  # 确保有足够时间释放端口
    log.info("ZMQ上下文已终止，ODrive 进程关闭")


def main():
    self_check("ch100", "odrive")  # 进行自检

    ch100_thread = threading.Thread(target=run_ch100_process)
    ch100_thread.start()

    odrive_thread = threading.Thread(target=run_odrive_process)
    odrive_thread.start()

    try:
        while True:
            time.sleep(1)  # 主线程保持运行
    except KeyboardInterrupt:
        log.info("主线程被用户中断")
        stop_event.set()  # 设置事件以停止子线程
    finally:
        ch100_thread.join()
        log.info("CH100 线程已结束")
        odrive_thread.join()
        log.info("ODrive 线程已结束")


if __name__ == "__main__":
    main()
