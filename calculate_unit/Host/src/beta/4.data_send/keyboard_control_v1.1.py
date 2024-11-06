import socket
import struct
import keyboard  # 需要安装keyboard库

# 常量定义
FRAME_HEADER = 0x30  # 帧头
FRAME_FOOTER = 0x40  # 帧尾
CRC8_INIT = 0x00     # CRC-8初始值

def crc8(data: bytes) -> int:
    """计算CRC-8校验值"""
    crc = CRC8_INIT
    for byte in data:
        crc ^= byte
    return crc

def send_integer(tcp_socket, value):
    """通过TCP发送16位整数"""
    # 确保值在16位整数范围内
    if not (-32768 <= value <= 32767):
        print("输入值超出范围！")
        return

    # 打包数据
    data_bytes = struct.pack('<h', value)  # 小端格式的16位整数
    crc = crc8([FRAME_HEADER] + list(data_bytes))  # 计算CRC
    frame = bytearray([FRAME_HEADER]) + data_bytes + bytearray([crc, FRAME_FOOTER])  # 构造完整帧
    
    # 发送数据
    tcp_socket.sendall(frame)
    print(f"发送帧: {list(frame)}")

def main():
    host = '192.168.2.171'  # Jetson平台的IP地址
    port = 12345            # 服务器端口，根据实际情况修改

    # 创建TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
        tcp_socket.connect((host, port))  # 连接到服务器
        print("已连接到服务器。按下 'a' 发送 +30，按下 'd' 发送 -30，按 'q' 退出。")

        while True:
            if keyboard.is_pressed('a'):  # 检测按键 'a'
                send_integer(tcp_socket, 30)  # 发送 +30
            elif keyboard.is_pressed('d'):  # 检测按键 'd'
                send_integer(tcp_socket, -30)  # 发送 -30
            elif keyboard.is_pressed('q'):  # 检测按键 'q'
                print("退出程序。")
                break  # 退出循环

if __name__ == "__main__":
    main()
