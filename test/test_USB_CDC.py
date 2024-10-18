import serial
import time

# 替换为你的 USB CDC 设备的串口号
USB_CDC_PORT = "COM10"  # 在 Windows 上通常是 COM3，Linux 上可能是 /dev/ttyUSB0
BAUD_RATE = 460800  # 根据你的设备设置波特率


def main():
    try:
        # 打开串口
        ser = serial.Serial(USB_CDC_PORT, BAUD_RATE, timeout=1)
        print(f"连接到 {USB_CDC_PORT}，波特率：{BAUD_RATE}")

        # 等待设备准备就绪
        time.sleep(2)

        # 发送测试消息
        test_message = "v 0 10\n"
        ser.write(test_message.encode())
        print(f"发送: {test_message.strip()}")

        # 接收响应
        while True:
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"接收: {response}")

    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except KeyboardInterrupt:
        print("程序被中断")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")


if __name__ == "__main__":
    main()
