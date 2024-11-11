import serial
import struct
import logging

# 常量定义
CHSYNC1 = 0x5A
CHSYNC2 = 0xA5
CH_HDR_SIZE = 6
FRAME_TAG_HI91 = 0x91


class CH100Frame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frame_type = None
        self.temperature = None
        self.pressure = None
        self.system_time_ms = None
        self.acc = None
        self.gyr = None
        self.mag = None
        self.quat = None
        self.sync_time = None
        self.roll = None
        self.pitch = None
        self.yaw = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class CH100Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.frame = CH100Frame()

    @staticmethod
    def crc16_update(crc, data):
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                temp = crc << 1
                if crc & 0x8000:
                    temp ^= 0x1021
                crc = temp
        return crc & 0xFFFF

    def _parse_hi91(self, data, ofs):
        self.frame.sync_time = struct.unpack_from("<H", data, ofs + 1)[0] * 1e-3
        self.frame.temperature = struct.unpack_from("<b", data, ofs + 3)[0]
        self.frame.pressure = struct.unpack_from("<f", data, ofs + 4)[0]
        self.frame.system_time_ms = struct.unpack_from("<I", data, ofs + 8)[0]
        self.frame.acc = struct.unpack_from("<3f", data, ofs + 12)
        self.frame.gyr = struct.unpack_from("<3f", data, ofs + 24)
        self.frame.mag = struct.unpack_from("<3f", data, ofs + 36)
        self.frame.roll = struct.unpack_from("<f", data, ofs + 48)[0]
        self.frame.pitch = struct.unpack_from("<f", data, ofs + 52)[0]
        self.frame.yaw = struct.unpack_from("<f", data, ofs + 56)[0]
        self.frame.quat = struct.unpack_from("<4f", data, ofs + 60)

    def parse(self, new_data):
        self.buffer += new_data
        frames = []
        while len(self.buffer) >= CH_HDR_SIZE:
            if self.buffer[0] == CHSYNC1 and self.buffer[1] == CHSYNC2:
                length = struct.unpack_from("<H", self.buffer, 2)[0]
                total_length = CH_HDR_SIZE + length
                if len(self.buffer) >= total_length:
                    frame_data = self.buffer[:total_length]
                    crc_calculated = self.crc16_update(
                        0, frame_data[:4] + frame_data[6:]
                    )
                    crc_received = struct.unpack_from("<H", frame_data, 4)[0]
                    if crc_calculated == crc_received:
                        self.frame.reset()
                        self.frame.frame_type = frame_data[6]
                        if self.frame.frame_type == FRAME_TAG_HI91:
                            self._parse_hi91(frame_data[CH_HDR_SIZE:], 0)
                            frames.append(self.frame.to_dict())
                        else:
                            logging.info(f"跳过帧类型: {self.frame.frame_type}")
                    else:
                        logging.error("CRC 校验失败")
                    del self.buffer[:total_length]
                else:
                    break
            else:
                del self.buffer[0]
        return frames


class CH100Device:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.parser = CH100Parser()

    def open(self):
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1)
        # logging.info(f"打开串口 {self.port}，波特率 {self.baudrate}")

    def close(self):
        if self.serial:
            self.serial.close()
            # logging.info(f"关闭串口 {self.port}")

    def read_and_parse(self):
        data = self.serial.read(82)  # CORE
        if data:
            frames = self.parser.parse(data)
            return frames
        return []
