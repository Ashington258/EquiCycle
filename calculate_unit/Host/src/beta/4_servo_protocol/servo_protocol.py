import serial
import struct
import logging

# 常量定义
FRAME_HEADER = 0x30  # 帧头
FRAME_FOOTER = 0x40  # 帧尾
CRC8_INIT = 0x00

class Int16Frame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None  # 16位数据值

    def to_dict(self):
        return {'value': self.value}

class Int16Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.frame = Int16Frame()

    @staticmethod
    def crc8(data: bytes) -> int:
        """简单的CRC-8计算函数"""
        crc = CRC8_INIT
        for byte in data:
            crc ^= byte
        return crc

    def parse(self, new_data):
        """解析数据流并提取完整帧"""
        self.buffer += new_data
        frames = []
        
        while len(self.buffer) >= 5:  # 数据帧至少5字节
            if self.buffer[0] == FRAME_HEADER and self.buffer[4] == FRAME_FOOTER:
                # 检查CRC并解析数据
                data_bytes = self.buffer[1:3]
                if len(data_bytes) == 2:
                    self.frame.reset()
                    self.frame.value = struct.unpack('<h', data_bytes)[0]  # 转换为16位整数

                    # CRC计算
                    crc_calculated = self.crc8(self.buffer[:3])  # 计算帧头和数据的CRC
                    crc_received = self.buffer[3]

                    if crc_calculated == crc_received:
                        frames.append(self.frame.to_dict())
                    else:
                        logging.error("CRC 校验失败")
                    
                    # 移除已处理的帧
                    del self.buffer[:5]  # 移除帧头、数据、CRC和帧尾
                else:
                    break
            else:
                del self.buffer[0]  # 移除不匹配的帧头
        
        return frames

class Int16Device:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.parser = Int16Parser()

    def open(self):
        """打开串口"""
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1)
        logging.info(f"打开串口 {self.port}，波特率 {self.baudrate}")

    def close(self):
        """关闭串口"""
        if self.serial:
            self.serial.close()
            logging.info(f"关闭串口 {self.port}")

    def read_and_parse(self):
        """读取并解析数据"""
        data = self.serial.read(5)  # 读取一帧数据（5字节）
        if data:
            frames = self.parser.parse(data)
            return frames
        return []
