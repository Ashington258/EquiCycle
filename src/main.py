import threading
import time
import json
import logging
from self_check.self_check import self_check
from ch100_protocol.ch100_protocol import CH100Device
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol
from ZMQ_Publisher.publisher import ZMQPublisher

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"
)


def load_config(file_path="src/config.json"):
    with open(file_path, "r") as f:
        return json.load(f)


config = load_config()

# 设备配置
ch100_port = config["ch100"]["port"]
ch100_baudrate = config["ch100"]["baudrate"]
ch100_zmq_port = config["ch100"]["zmq_port"]
odrive_port = config["odrive"]["port"]
odrive_baudrate = config["odrive"]["baudrate"]
odrive_zmq_port = config["odrive"]["zmq_port"]

# 全局停止事件
stop_event = threading.Event()


def run_ch100_process():
    logging.info("CH100 线程启动")
    ch100_device = CH100Device(port=ch100_port, baudrate=ch100_baudrate)
    ch100_device.open()
    publisher = ZMQPublisher(port=ch100_zmq_port)

    try:
        while not stop_event.is_set():
            frames = ch100_device.read_and_parse()
            for frame in frames:
                publisher.send_json(frame)
                logging.info(f"已发布帧: {frame}")
    except Exception as e:
        logging.error(f"CH100 进程发生错误: {e}")
    finally:
        ch100_device.close()
        publisher.close()
        logging.info("CH100 进程已终止")


def run_odrive_process():
    logging.info("ODrive 线程启动")
    odrive = ODriveAsciiProtocol(port=odrive_port, baudrate=odrive_baudrate)
    publisher = ZMQPublisher(port=odrive_zmq_port)

    try:
        while not stop_event.is_set():
            try:
                odrive.motor_velocity(0, 8)
                feedback = odrive.request_feedback(0)
                publisher.send_string(f"motor_speed {feedback}")
                logging.info(f"已发布 ODrive 反馈: {feedback}")
            except Exception as e:
                logging.error(f"ODrive 进程发生错误: {e}")
                break
    except Exception as e:
        logging.error(f"ODrive 进程发生错误: {e}")
    finally:
        odrive.close()
        publisher.close()
        logging.info("ODrive 进程已终止")


def main():

    self_check("ch100", "odrive")

    ch100_thread = threading.Thread(target=run_ch100_process, name="CH100Thread")
    odrive_thread = threading.Thread(target=run_odrive_process, name="ODriveThread")

    ch100_thread.start()
    odrive_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("主线程被用户中断")
        stop_event.set()
    finally:
        ch100_thread.join()
        logging.info("CH100 线程已结束")
        odrive_thread.join()
        logging.info("ODrive 线程已结束")


if __name__ == "__main__":
    main()
