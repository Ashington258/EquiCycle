import socket


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


# 示例使用
if __name__ == "__main__":
    target = 1  # 控制目标
    speed = 100  # 速度值
    ip_address = "192.168.1.100"  # 目标 IP 地址
    port_number = 12345  # 目标端口号

    sender = ControlFlowSender(ip_address, port_number)
    sender.send_udp_message(target, speed)