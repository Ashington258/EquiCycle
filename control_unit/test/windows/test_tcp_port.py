import json
import socket
import time


def load_config_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def check_zmq_port_availability(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)  # 设置超时时间为1秒
        try:
            s.bind(("", port))  # 尝试绑定到端口
            return True
        except socket.error:
            return False


def main():
    json_file_path = "src/config.json"  # JSON文件路径
    config = load_config_from_json(json_file_path)

    for device, settings in config.items():
        zmq_port = settings["zmq_port"]
        if check_zmq_port_availability(zmq_port):
            print(f"ZMQ port {zmq_port} for {device} is available.")
        else:
            print(f"ZMQ port {zmq_port} for {device} is not available.")


if __name__ == "__main__":
    main()
