# control_unit/src/directional_control/directional_control.py

import threading


class DirectionalControl:
    def __init__(self):
        self.pulse_value = None
        self.lock = threading.Lock()  # 用于线程安全

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
