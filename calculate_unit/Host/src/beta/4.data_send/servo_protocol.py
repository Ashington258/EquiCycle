# -*- coding: utf-8 -*-

import serial
import struct
import logging
import socket

# 常量定义
FRAME_HEADER = 0x30  # 帧头
FRAME_FOOTER = 0x40  # 帧尾
CRC8_INIT = 0x00

def crc8(data: bytes) -> int:
    """计算CRC-8校验值"""
    crc = 0x00
    for byte in data:
        crc ^= byte
    return crc

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

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 12345))  # 下位机运行端口
    server_socket.listen(1)
    print("等待连接...")
    
    conn, addr = server_socket.accept()
    with conn:
        print(f"已连接到 {addr}")
        buffer = bytearray()
        
        while True:
            data = conn.recv(1024)  # 接收数据
            if not data:
                break
            
            buffer.extend(data)  # 将接收到的数据添加到缓冲区
            print(f"接收到的原始数据: {list(data)}")  # 调试输出
            
            while True:
                # 查找帧头和帧尾
                header_index = buffer.find(bytes([FRAME_HEADER]))
                footer_index = buffer.find(bytes([FRAME_FOOTER]), header_index)
                
                if header_index != -1 and footer_index != -1 and footer_index > header_index:
                    # 提取有效数据帧
                    frame = buffer[header_index:footer_index + 1]
                    data_bytes = frame[1:-2]  # 去掉帧头和帧尾，保留CRC
                    crc_received = frame[-2]  # 接收到的CRC
                    buffer = buffer[footer_index + 1:]  # 移除已处理的帧
                    
                    # 计算CRC并验证
                    crc_calculated = crc8(frame[:-2])  # 计算CRC
                    if crc_received == crc_calculated:
                        if len(data_bytes) == 2:
                            value = struct.unpack('<h', data_bytes)[0]  # 解包16位整数
                            print(f"接收到的值: {value}")
                        else:
                            print(f"接收到的无效数据长度: {len(data_bytes)}")
                    else:
                        print("CRC校验失败！")
                else:
                    break

if __name__ == "__main__":
    main()