import socket
import threading
import serial


class ControlFlowSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def create_control_flow(self, target, speed):
        # 确保速度在合理范围内
        if not isinstance(speed, int) or speed < 0 or speed > 255:
            raise ValueError("Speed must be an integer between 0 and 255.")

        # 将速度转换为 ASCII 字符
        speed_ascii = str(speed)

        # 构建控制流字符串
        control_flow = f"v {target} {speed_ascii}\n"
        return control_flow

    def send_udp_message(self, target, speed):
        # 创建控制流字符串
        control_flow_string = self.create_control_flow(target, speed)

        # 创建 UDP 套接字
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # 发送数据
            sock.sendto(control_flow_string.encode("utf-8"), (self.ip, self.port))
            print(f"Sent: {control_flow_string.strip()} to {self.ip}:{self.port}")


class ControlFlowReceiver:
    def __init__(self, ip, port, serial_port, baudrate):
        self.ip = ip
        self.port = port
        self.serial_port = serial.Serial(serial_port, baudrate, timeout=1)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

    def start_receiving(self):
        print(f"Listening for UDP messages on {self.ip}:{self.port}")
        while True:
            data, addr = self.sock.recvfrom(1024)  # Buffer size is 1024 bytes
            message = data.decode("utf-8")
            print(f"Received message: {message} from {addr}")
            self.send_to_serial(message)

    def send_to_serial(self, message):
        # 发送数据到串口
        self.serial_port.write(message.encode("utf-8"))
        print(f"Sent to serial: {message.strip()}")


# 示例使用
if __name__ == "__main__":
    # 启动接收器
    receiver = ControlFlowReceiver(
        "0.0.0.0", 12345, "/dev/ttyUSB0", 9600
    )  # 修改串口号和波特率
    threading.Thread(target=receiver.start_receiving, daemon=True).start()

    # 启动发送器
    target = 1  # 控制目标
    speed = 100  # 速度值
    ip_address = "192.168.1.100"  # 目标 IP 地址
    port_number = 12345  # 目标端口号

    sender = ControlFlowSender(ip_address, port_number)
    sender.send_udp_message(target, speed)

    # 让主线程保持运行，以便接收消息
    input("Press Enter to exit...\n")
