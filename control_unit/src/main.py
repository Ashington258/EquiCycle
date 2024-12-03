import threading
import time
import json
import logging
import queue
from self_check.self_check import self_check
from ch100_protocol.ch100_protocol import CH100Device
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol
from balance_control.balance_control import control_layer  # 导入 control_layer 函数
from directional_control.directional_control import (
    DirectionalControl,
)  # 导入 DirectionalControl 类
import socket
import serial

from balance_control.balance_control import process_speedBack_message  # 导入 process_speedBack_message 中的函数
from balance_control.balance_control import process_steer_dynamicAngle
# 移除所有的日志处理程序，关闭日志模块
logging.getLogger().handlers.clear()


# 加载配置
def load_config(file_path="/home/jetson/workspace/EquiCycle/control_unit/src/config.json"):
    with open(file_path, "r") as f:
        return json.load(f)


config = load_config()

# 设备配置
ch100_config = config["ch100"]
odrive_config = config["odrive"]
servo_config = config["servo"]

# 初始化串口
ser = serial.Serial(servo_config["port"], servo_config["baudrate"])

# 全局停止事件
stop_event = threading.Event()

# 用于线程间通信的队列
data_queue = queue.Queue()

# 创建 DirectionalControl 实例
directional_control = DirectionalControl()


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
            except Exception as e:
                logging.error(f"ODrive 线程错误: {e}")
            time.sleep(0.005)  # 限制请求速率为 5ms
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
                data = data_queue.get(timeout=1)
                control_layer(data, odrive_instance)  # 将数据传递给控制层
            except queue.Empty:
                pass  # 超时发生，继续循环
            except Exception as e:
                logging.error(f"控制线程错误: {e}")
    finally:
        logging.info("控制线程已停止")


def servo_listener():
    """
    UDP 监听线程，用于接收数据并解析脉冲值
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((servo_config["udp_host"], servo_config["udp_port"]))
        print("从机端已启动，等待主机端数据...")

        while not stop_event.is_set():
            data, addr = sock.recvfrom(1024)
            if data:
                # 解析数据协议并更新脉冲值
                pulse_value = directional_control.parse_protocol(data)
                if pulse_value is not None:
                    # 将数据转发到串口
                    ser.write(data)
                    process_steer_dynamicAngle(pulse_value)
                    print("接收到的数据帧:", [hex(x) for x in data])
                    
def speed_BackWheel_listener():
    """
    新增的UDP监听线程，用于接收来自另一个端口的指令
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("0.0.0.0", 12345))  # 监听端口 12345
        print("视觉监听线程已启动，等待控制指令...")

        while not stop_event.is_set():
            data, addr = sock.recvfrom(1024)
            if data:
                # 假设收到的数据格式是 "v <target> <speed>"
                control_message = data.decode("utf-8").strip()
                print(f"接收到视觉的控制指令: {control_message}")

                # 调用 balance_control 中的函数来处理控制消息
                try:
                    process_speedBack_message(control_message)
                except Exception as e:
                    logging.error(f"处理控制指令时发生错误: {e}")
            else:
                process_speedBack_message(0)

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
        threading.Thread(
            target=servo_listener, name="ServoListener"
        ),  # 新增的 servo_listener 线程
        threading.Thread(
            target=speed_BackWheel_listener,
            name="SpeedBackWheelListener",
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
