import socket
import struct
import keyboard  # 需要安装keyboard库
import sys

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
    if not (-32768 <= value <= 32767):
        print("输入值超出范围！")
        return

    data_bytes = struct.pack('<h', value)  # 小端格式的16位整数
    crc = crc8([FRAME_HEADER] + list(data_bytes))  # 计算CRC
    frame = bytearray([FRAME_HEADER]) + data_bytes + bytearray([crc, FRAME_FOOTER])  # 构造完整帧
    
    tcp_socket.sendall(frame)
    print(f"发送帧: {list(frame)}")

def increase_value():
    global value
    value += delta
    send_integer(tcp_socket, value)

def decrease_value():
    global value
    value -= delta
    send_integer(tcp_socket, value)

def main():
    global tcp_socket, value, delta
    host = '192.168.2.159'  # Jetson平台的IP地址
    port = 12345            # 服务器端口，根据实际情况修改
    value = 1350            # 初始值
    delta = 25              # 每次变化的量

    # 创建TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
        tcp_socket.connect((host, port))  # 连接到服务器
        print("已连接到服务器。按下 'a' 增加值，按下 'd' 减少值，按 'q' 退出。")

        # 设置按键钩子
        keyboard.hook(lambda e: None)  # 捕获所有按键事件并不处理，避免显示

        keyboard.add_hotkey('a', increase_value)
        keyboard.add_hotkey('d', decrease_value)

        # 处理退出逻辑
        def exit_program():
            print("\n程序已退出。")
            keyboard.unhook_all()  # 解除所有钩子
            # os.system('cls' if os.name == 'nt' else 'clear')  # 清屏（可选）
            sys.exit(0)

        keyboard.add_hotkey('q', exit_program)  # 退出程序的钩子

        # 保持程序运行
        print("程序运行中，按 'q' 退出。")
        keyboard.wait('q')  # 等待 'q' 键被按下

if __name__ == "__main__":
    main()
