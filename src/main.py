import threading
import time
import json
import logging
import zmq
from self_check.self_check import self_check
from ch100_protocol.ch100_protocol import CH100Device
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol
from ZMQ_Publisher.publisher import ZMQPublisher
from balance_control.balance_control import (
    control_layer,
)  # 导入控制模块中的 control_layer 函数

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"
)


def load_config(file_path="src/config.json"):
    with open(file_path, "r") as f:
        return json.load(f)


config = load_config()

# Device configuration
ch100_config = config["ch100"]
odrive_config = config["odrive"]

# Global stop event
stop_event = threading.Event()


def setup_publisher(port):
    publisher = ZMQPublisher(port=port)
    return publisher


def ch100_thread_function():
    logging.info("Starting CH100 thread")
    ch100_device = CH100Device(
        port=ch100_config["port"], baudrate=ch100_config["baudrate"]
    )
    ch100_device.open()
    publisher = setup_publisher(ch100_config["zmq_port"])

    try:
        while not stop_event.is_set():
            frames = ch100_device.read_and_parse()
            for frame in frames:
                frame["device"] = "ch100"  # Add device identifier
                publisher.send_json(frame)
                logging.debug(f"Published CH100 frame: {frame}")
    except Exception as e:
        logging.error(f"CH100 thread error: {e}")
    finally:
        ch100_device.close()
        publisher.close()
        logging.info("CH100 thread stopped")


def odrive_thread_function():
    logging.info("启动 ODrive 线程")
    odrive = ODriveAsciiProtocol(
        port=odrive_config["port"], baudrate=odrive_config["baudrate"]
    )
    publisher = setup_publisher(odrive_config["zmq_port"])

    try:
        while not stop_event.is_set():
            try:
                feedback = odrive.request_feedback(0)
                data = {
                    "device": "odrive",  # 添加设备标识符
                    "feedback": feedback,
                }
                publisher.send_json(data)
                logging.info(f"已发布 ODrive 反馈: {data}")
            except Exception as e:
                logging.error(f"ODrive 线程错误: {e}")
    finally:
        odrive.close()
        publisher.close()
        logging.info("ODrive 线程已停止")


def zmq_monitoring_thread_function():
    logging.info("Starting monitoring thread")
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(f"tcp://localhost:{ch100_config['zmq_port']}")
    subscriber.connect(f"tcp://localhost:{odrive_config['zmq_port']}")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    try:
        while not stop_event.is_set():
            socks = dict(poller.poll(100))
            if subscriber in socks and socks[subscriber] == zmq.POLLIN:
                message = subscriber.recv_string()
                data = json.loads(message)

                # 将数据传递给控制层进行解析和处理
                control_layer(data)
    except Exception as e:
        logging.error(f"Monitoring thread error: {e}")
    finally:
        subscriber.close()
        context.term()
        logging.info("Monitoring thread stopped")


def main():
    self_check("ch100", "odrive")

    threads = [
        threading.Thread(target=ch100_thread_function, name="CH100Thread"),
        threading.Thread(target=odrive_thread_function, name="ODriveThread"),
        threading.Thread(
            target=zmq_monitoring_thread_function, name="MonitoringThread"
        ),
    ]

    for thread in threads:
        thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Main thread interrupted by user")
        stop_event.set()
    finally:
        for thread in threads:
            thread.join()
        logging.info("All threads have stopped")


if __name__ == "__main__":
    main()
