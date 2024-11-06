import socket
import struct

FRAME_HEADER = 0x30
FRAME_FOOTER = 0x40

def crc8(data: bytes) -> int:
    """计算CRC-8校验值"""
    crc = 0x00
    for byte in data:
        crc ^= byte
    return crc

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 12345))
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
