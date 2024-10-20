import struct
import logging
import serial
import zmq
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constant definitions
CHSYNC1 = 0x5A
CHSYNC2 = 0xA5
CH_HDR_SIZE = 6

GRAVITY = 9.80665
R2D = 57.29577951308232

# Data item identifiers
FRAME_TAG_HI91 = 0x91


class hipnuc_frame:
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
        """Return a dictionary representation of non-null fields"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class hipnuc_parser:
    def __init__(self):
        self.CHSYNC1 = CHSYNC1
        self.CHSYNC2 = CHSYNC2
        self.CH_HDR_SIZE = CH_HDR_SIZE
        self.buffer = bytearray()
        self.frame = hipnuc_frame()

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
        """Decode new data and return successfully parsed frames"""
        self.buffer += new_data
        frames = []
        while len(self.buffer) >= self.CH_HDR_SIZE:
            # Look for frame header
            if self.buffer[0] == self.CHSYNC1 and self.buffer[1] == self.CHSYNC2:
                length = struct.unpack_from("<H", self.buffer, 2)[0]
                if len(self.buffer) >= self.CH_HDR_SIZE + length:
                    frame_data = self.buffer[: self.CH_HDR_SIZE + length]
                    crc_calculated = self.crc16_update(
                        0, frame_data[:4] + frame_data[6:]
                    )
                    crc_received = struct.unpack_from("<H", frame_data, 4)[0]
                    if crc_calculated == crc_received:
                        self.frame.reset()  # Reset data
                        self.frame.frame_type = frame_data[6]
                        # Only parse HI91 frames
                        if self.frame.frame_type == FRAME_TAG_HI91:
                            self._parse_hi91(frame_data[self.CH_HDR_SIZE :], 0)
                            frames.append(self.frame.to_dict())
                        else:
                            logging.info(f"Skipped frame type: {self.frame.frame_type}")
                    else:
                        logging.error("CRC check failed")
                    del self.buffer[: self.CH_HDR_SIZE + length]
                else:
                    break
            else:
                # Remove invalid data
                del self.buffer[0]
        return frames


# Main code
if __name__ == "__main__":
    decoder = hipnuc_parser()

    # Set up serial port (请根据实际情况修改串口号和波特率)
    ser = serial.Serial(port="COM22", baudrate=460800, timeout=0.1)

    # Set up ZeroMQ publisher
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    logging.info("Starting data acquisition and publishing...")

    try:
        while True:
            # Read data from serial port
            data = ser.read(82)  # Read up to 1024 bytes
            if data:
                frames = decoder.parse(data)
                for frame in frames:
                    # Publish the parsed data via ZeroMQ
                    frame["topic"] = "imu_data"  # 添加主题信息
                    publisher.send_json(frame)
                    logging.info(f"Published frame: {frame}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        ser.close()
        publisher.close()
        context.term()
        logging.info("Shutting down")
