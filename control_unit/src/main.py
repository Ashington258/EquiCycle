import threading
import time
import json
import logging
import queue
from self_check.self_check import self_check
from ch100_protocol.ch100_protocol import CH100Device
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol
from balance_control.balance_control import control_layer  # 导入 control_layer 函数

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"
)


def load_config(file_path="control_unit/src/config.json"):
    with open(file_path, "r") as f:
        return json.load(f)


config = load_config()

# 设备配置
ch100_config = config["ch100"]
odrive_config = config["odrive"]

# 全局停止事件
stop_event = threading.Event()

# 用于线程间通信的队列
data_queue = queue.Queue()


def ch100_thread_function(data_queue):
    """
    CH100 线程从 CH100 串口设备读取数据并将其放入队列中。
    """
    logging.info("启动 CH100 线程")
    ch100_device = CH100Device(
        port=ch100_config["port"], baudrate=ch100_config["baudrate"]
    )
    ch100_device.open()

    try:
        while not stop_event.is_set():
            frames = ch100_device.read_and_parse()
            for frame in frames:
                frame["device"] = "ch100"  # 添加设备标识符
                data_queue.put(frame)  # 将帧放入队列
                logging.debug(f"已将 CH100 帧添加到队列: {frame}")
    except Exception as e:
        logging.error(f"CH100 线程错误: {e}")
    finally:
        ch100_device.close()
        logging.info("CH100 线程已停止")


def odrive_thread_function(odrive_instance, data_queue):
    """
    ODrive 线程从 ODrive 电机控制器读取反馈并将其放入队列中。
    """
    logging.info("启动 ODrive 线程")

    try:
        while not stop_event.is_set():
            try:
                feedback = odrive_instance.request_feedback(0)
                data = {
                    "device": "odrive",  # 添加设备标识符
                    "feedback": feedback,
                }
                data_queue.put(data)  # 将反馈放入队列
                logging.info(f"已将 ODrive 反馈添加到队列: {data}")
            except Exception as e:
                logging.error(f"ODrive 线程错误: {e}")
            time.sleep(0.01)  # 限制请求速率为 10ms
    finally:
        odrive_instance.close()
        logging.info("ODrive 线程已停止")


def control_thread_function(odrive_instance, data_queue):
    """
    控制线程从队列中读取数据并将其传递给控制层进行处理。
    """
    logging.info("启动控制线程")

    try:
        while not stop_event.is_set():
            try:
                # 从队列获取数据（在有项目可用时阻塞）
                data = data_queue.get(timeout=1)
                control_layer(data, odrive_instance)  # 将数据传递给控制层
                logging.info(f"已处理数据: {data}")
            except queue.Empty:
                pass  # 超时发生，继续循环
            except Exception as e:
                logging.error(f"控制线程错误: {e}")
    finally:
        logging.info("控制线程已停止")


def main():
    # 在启动线程之前执行系统自检
    self_check("ch100", "odrive")

    # 初始化 ODrive 实例
    odrive_instance = ODriveAsciiProtocol(
        port=odrive_config["port"], baudrate=odrive_config["baudrate"]
    )

    # 创建线程
    threads = [
        threading.Thread(
            target=ch100_thread_function, args=(data_queue,), name="CH100Thread"
        ),
        threading.Thread(
            target=odrive_thread_function,
            args=(odrive_instance, data_queue),
            name="ODriveThread",
        ),
        threading.Thread(
            target=control_thread_function,
            args=(odrive_instance, data_queue),
            name="ControlThread",
        ),
    ]

    # 启动所有线程
    for thread in threads:
        thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("主线程被用户中断")
        stop_event.set()
    finally:
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        logging.info("所有线程已停止")


if __name__ == "__main__":
    main()
