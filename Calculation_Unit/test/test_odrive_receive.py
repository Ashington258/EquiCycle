import socket
import threading
import serial


class ControlFlowSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def motor_velocity(self, motor: int, velocity: float, torque_ff: float = 0.0):
        # 构建命令字符串
        command = f"v {motor} {velocity}\n"
        self.send_command(command)

    def send_command(self, command: str):
        # 创建 UDP 套接字并发送命令
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(command.encode("utf-8"), (self.ip, self.port))
            print(f"Sent: {command.strip()} to {self.ip}:{self.port}")


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
    receiver = ControlFlowReceiver("0.0.0.0", 5000, "COM20", 9600)  # 修改串口号和波特率
    threading.Thread(target=receiver.start_receiving, daemon=True).start()

    # 启动发送器
    target_motor = 1  # 控制目标
    velocity = 1.34  # 速度值，可以是小数
    ip_address = "192.168.2.36"  # 目标 IP 地址
    port_number = 12345  # 目标端口号

    sender = ControlFlowSender(ip_address, port_number)
    sender.motor_velocity(target_motor, velocity)

    # 让主线程保持运行，以便接收消息
    input("Press Enter to exit...\n")
