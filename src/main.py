import threading
from ch100_protocol.hipnuc_serial_parser_zmq import hipnuc_parser
from odrive_protocol.odrive_protocol_zmq import ODriveAsciiProtocol
from self_check.self_check import self_check
import serial
import zmq
import logging
import time
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 创建一个全局事件
stop_event = threading.Event()


def run_ch100_process(port="COM22", baudrate=460800):
    decoder = hipnuc_parser()
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)

    while True:
        try:
            publisher.bind("tcp://*:5555")
            break
        except zmq.error.ZMQError as e:
            logging.error(f"绑定失败: {e}，重试中...")
            time.sleep(1)

    logging.info("开始 CH100 数据采集和发布...")

    try:
        while not stop_event.is_set():  # 检查事件
            data = ser.read(1024)
            if data:
                frames = decoder.parse(data)
                for frame in frames:
                    publisher.send_json(frame)
                    logging.info(f"已发布帧: {frame}")
    except Exception as e:
        logging.error(f"发生错误: {e}")
    finally:
        ser.close()
        publisher.close()
        context.term()
        logging.info("CH100 进程关闭")


def run_odrive_process(port="COM24", baudrate=460800):
    odrive = ODriveAsciiProtocol(port=port, baudrate=baudrate)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)

    while True:
        try:
            publisher.bind("tcp://*:5556")
            break
        except zmq.error.ZMQError as e:
            logging.error(f"绑定失败: {e}，重试中...")
            time.sleep(1)

    while not stop_event.is_set():  # 检查事件
        try:
            odrive.motor_velocity(0, 10)
            odrive.request_feedback(0)
            feedback = odrive.request_feedback(0)
            print(f"电机反馈: {feedback}")
            publisher.send_string(f"motor_speed {feedback}")
        except Exception as e:
            logging.error(f"ODrive 进程发生错误: {e}")
            break
    odrive.close()


def main():
    self_check()  # 进行自检

    ch100_thread = threading.Thread(target=run_ch100_process)
    ch100_thread.start()

    odrive_thread = threading.Thread(target=run_odrive_process)
    odrive_thread.start()

    try:
        while True:
            time.sleep(1)  # 主线程保持运行
    except KeyboardInterrupt:
        logging.info("主线程被用户中断")
        stop_event.set()  # 设置事件以停止子线程
    finally:
        ch100_thread.join()
        odrive_thread.join()


if __name__ == "__main__":
    main()
