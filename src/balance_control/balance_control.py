import zmq
import json
import ast
import re


# 从 JSON 文件读取设备配置
def load_device_config(filename="src/config.json"):
    with open(filename, "r") as f:
        return json.load(f)


def pull_messages(config):
    # 创建 ZMQ 上下文
    context = zmq.Context()

    # 创建一个拉取器（PULL）来接收消息
    sockets = []
    ports = []

    # 提取 ZMQ 端口
    for device in config.values():
        port = device["zmq_port"]
        socket = context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{port}")
        sockets.append(socket)
        ports.append(port)

    imu_data = {}
    motor_speed = None

    try:
        while True:
            # 接收消息
            for socket in sockets:
                try:
                    message = socket.recv_string(flags=zmq.NOBLOCK)  # 非阻塞接收
                    port = (
                        socket.getsockopt(zmq.LAST_ENDPOINT).decode().split(":")[-1]
                    )  # 获取端口

                    # 解析消息
                    if port == "5557":  # ch100 IMU 数据
                        # 使用正则表达式提取帧数据
                        match = re.search(r"已发布帧: (.+)", message)
                        if match:
                            frame_data = ast.literal_eval(match.group(1))
                            imu_data["acc"] = frame_data["acc"]
                            imu_data["gyr"] = frame_data["gyr"]
                            imu_data["roll"] = frame_data["roll"]
                            imu_data["pitch"] = frame_data["pitch"]
                            imu_data["yaw"] = frame_data["yaw"]
                            print(f"🎃IMU 数据: {imu_data}")

                    elif port == "5558":  # odrive 电机转速
                        # 使用正则表达式提取电机转速
                        match = re.search(
                            r"已发布 ODrive 反馈: ([\d.-]+) ([\d.-]+)", message
                        )
                        if match:
                            motor_speed = float(
                                match.group(1)
                            )  # 假设第一个数字是电机转速
                            print(f"🎈电机转速: {motor_speed}")

                except zmq.Again:
                    # 没有消息可接收，继续循环
                    continue

    except KeyboardInterrupt:
        print("停止消息接收...")
    finally:
        for socket in sockets:
            socket.close()
        context.term()


if __name__ == "__main__":
    config = load_device_config()  # 读取设备配置
    pull_messages(config)  # 拉取消息
