import threading
from ch100_protocol.hipnuc_serial_parser_zmq import hipnuc_parser
from odrive_protocol.odrive_protocol_zmq import ODriveAsciiProtocol
import serial
import zmq
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_ch100_process(port="COM22", baudrate=460800):
    decoder = hipnuc_parser()

    # 设置串口
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)

    # 设置 ZeroMQ 发布者
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)

    # 尝试绑定发布者，直到成功
    while True:
        try:
            publisher.bind("tcp://*:5555")  # CH100 使用 5555
            break
        except zmq.error.ZMQError as e:
            logging.error(f"绑定失败: {e}，重试中...")
            time.sleep(1)

    logging.info("开始 CH100 数据采集和发布...")

    try:
        while True:
            # 从串口读取数据
            data = ser.read(1024)  # 读取最多 1024 字节
            if data:
                frames = decoder.parse(data)
                for frame in frames:
                    # 通过 ZeroMQ 发布解析后的数据
                    publisher.send_json(frame)
                    logging.info(f"已发布帧: {frame}")
    except KeyboardInterrupt:
        logging.info("被用户中断")
    except Exception as e:
        logging.error(f"发生错误: {e}")
    finally:
        ser.close()
        publisher.close()
        context.term()
        logging.info("CH100 进程关闭")


def run_odrive_process(port="COM24", baudrate=460800):
    # 初始化 ODrive 并使用适当的串口和波特率
    odrive = ODriveAsciiProtocol(port=port, baudrate=baudrate)

    # 设置 ZeroMQ 发布者
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)

    # 尝试绑定发布者，直到成功
    while True:
        try:
            publisher.bind("tcp://*:5556")  # ODrive 使用 5556
            break
        except zmq.error.ZMQError as e:
            logging.error(f"绑定失败: {e}，重试中...")
            time.sleep(1)

    while True:
        try:
            # 示例：设置电机速度
            odrive.motor_velocity(0, 10)
            odrive.request_feedback(0)

            # 获取电机反馈
            feedback = odrive.request_feedback(0)
            print(f"电机反馈: {feedback}")

            # 这里可以发布电机反馈数据
            publisher.send_string(f"motor_speed {feedback}")

        except KeyboardInterrupt:
            logging.info("ODrive 进程被用户中断")
            break
        except Exception as e:
            logging.error(f"ODrive 进程发生错误: {e}")
            break
        finally:
            odrive.close()


def main():
    # 在单独的线程中启动 CH100 进程
    ch100_thread = threading.Thread(target=run_ch100_process)
    ch100_thread.start()

    # 在单独的线程中启动 ODrive 进程
    odrive_thread = threading.Thread(target=run_odrive_process)
    odrive_thread.start()

    # 等待线程结束
    ch100_thread.join()
    odrive_thread.join()


if __name__ == "__main__":
    main()
