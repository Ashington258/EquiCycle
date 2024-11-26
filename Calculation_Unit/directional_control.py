import threading
import socket


class DirectionalControl:
    def __init__(self, udp_ip="127.0.0.1", udp_port=12345):
        self.pulse_value = None
        self.lock = threading.Lock()  # 用于线程安全
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建 UDP 套接字

    def parse_protocol(self, data):
        """
        解析协议并提取脉冲值
        """
        if len(data) != 5:
            print("接收到的数据长度不正确")
            return None

        # 检查帧头和帧尾
        if data[0] != 0x30 or data[4] != 0x40:
            print("帧头或帧尾不正确")
            return None

        # 提取并计算脉冲值（小端序）
        pulse_value = data[1] | (data[2] << 8)

        # 计算并验证 CRC
        crc_calculated = data[0] ^ data[1] ^ data[2]
        if crc_calculated != data[3]:
            print("CRC 校验失败")
            return None

        # 更新脉冲值
        with self.lock:
            self.pulse_value = pulse_value

        print(f"脉冲值解析成功: {pulse_value}")
        return pulse_value

    def get_pulse_value(self):
        """
        返回最新的脉冲值
        """
        with self.lock:
            return self.pulse_value

    def build_protocol_frame(self, pulse_width):
        """
        构建协议数据帧
        """
        # 帧头和帧尾
        frame_header = 0x30
        frame_footer = 0x40

        # 分离脉冲宽度的低字节和高字节
        low_byte = pulse_width & 0xFF
        high_byte = (pulse_width >> 8) & 0xFF

        # 计算 CRC
        crc = frame_header ^ low_byte ^ high_byte

        # 构建完整协议帧
        protocol_frame = [frame_header, low_byte, high_byte, crc, frame_footer]
        print(f"构建的协议帧: {protocol_frame} 脉冲宽度：{pulse_width}")
        return protocol_frame

    def send_protocol_frame_udp(self, pulse_width):
        """
        构建并通过 UDP 发送协议帧
        """
        protocol_frame = self.build_protocol_frame(pulse_width)

        # 将协议帧转换为字节数组
        protocol_frame_bytes = bytes(protocol_frame)

        # 通过 UDP 发送数据
        try:
            self.sock.sendto(protocol_frame_bytes, (self.udp_ip, self.udp_port))
            print(f"协议帧已通过 UDP 发送到 {self.udp_ip}:{self.udp_port}")
        except Exception as e:
            print(f"发送 UDP 数据时出错: {e}")
