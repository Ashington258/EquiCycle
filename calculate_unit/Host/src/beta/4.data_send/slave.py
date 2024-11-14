import serial
import struct
import logging
import socket

# Constant definitions
FRAME_HEADER = 0x30  # Frame header
FRAME_FOOTER = 0x40  # Frame footer
CRC8_INIT = 0x00

def crc8(data: bytes) -> int:
    """Calculate CRC-8 checksum"""
    crc = 0x00
    for byte in data:
        crc ^= byte
    return crc

class Int16Frame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None  # 16-bit data value

    def to_dict(self):
        return {'value': self.value}

class Int16Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.frame = Int16Frame()

    @staticmethod
    def crc8(data: bytes) -> int:
        """Simple CRC-8 calculation function"""
        crc = CRC8_INIT
        for byte in data:
            crc ^= byte
        return crc

    def parse(self, new_data):
        """Parse the data stream and extract complete frames"""
        self.buffer += new_data
        frames = []
        
        while len(self.buffer) >= 5:  # Data frame must be at least 5 bytes
            if self.buffer[0] == FRAME_HEADER and self.buffer[4] == FRAME_FOOTER:
                # Check CRC and parse data
                data_bytes = self.buffer[1:3]
                if len(data_bytes) == 2:
                    self.frame.reset()
                    self.frame.value = struct.unpack('<h', data_bytes)[0]  # Convert to 16-bit integer

                    # CRC calculation
                    crc_calculated = self.crc8(self.buffer[:3])  # Calculate CRC for header and data
                    crc_received = self.buffer[3]

                    if crc_calculated == crc_received:
                        frames.append(self.frame.to_dict())
                    else:
                        logging.error("CRC check failed")
                    
                    # Remove processed frame
                    del self.buffer[:5]  # Remove header, data, CRC, and footer
                else:
                    break
            else:
                del self.buffer[0]  # Remove unmatched frame header
        
        return frames

class Int16Device:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.parser = Int16Parser()

    def open(self):
        """Open the serial port"""
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1)
        logging.info(f"Opened serial port {self.port}, baudrate {self.baudrate}")

    def close(self):
        """Close the serial port"""
        if self.serial:
            self.serial.close()
            logging.info(f"Closed serial port {self.port}")

    def read_and_parse(self):
        """Read and parse data"""
        data = self.serial.read(5)  # Read one frame of data (5 bytes)
        if data:
            frames = self.parser.parse(data)
            return frames
        return []

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 12345))  # Port for the device to run
    server_socket.listen(1)
    print("Waiting for connection...")
    
    conn, addr = server_socket.accept()
    with conn:
        print(f"Connected to {addr}")
        buffer = bytearray()
        
        while True:
            data = conn.recv(1024)  # Receive data
            if not data:
                break
            
            buffer.extend(data)  # Add received data to the buffer
            print(f"Received raw data: {list(data)}")  # Debug output
            
            while True:
                # Find frame header and footer
                header_index = buffer.find(bytes([FRAME_HEADER]))
                footer_index = buffer.find(bytes([FRAME_FOOTER]), header_index)
                
                if header_index != -1 and footer_index != -1 and footer_index > header_index:
                    # Extract valid data frame
                    frame = buffer[header_index:footer_index + 1]
                    data_bytes = frame[1:-2]  # Remove header and footer, keep CRC
                    crc_received = frame[-2]  # Received CRC
                    buffer = buffer[footer_index + 1:]  # Remove processed frame
                    
                    # Calculate CRC and verify
                    crc_calculated = crc8(frame[:-2])  # Calculate CRC
                    if crc_received == crc_calculated:
                        if len(data_bytes) == 2:
                            value = struct.unpack('<h', data_bytes)[0]  # Unpack 16-bit integer
                            print(f"Received value: {value}")
                        else:
                            print(f"Received invalid data length: {len(data_bytes)}")
                    else:
                        print("CRC check failed!")
                else:
                    break

if __name__ == "__main__":
    main()
