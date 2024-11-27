import serial.tools.list_ports


def list_usb_devices():
    # 获取所有可用的串口
    ports = serial.tools.list_ports.comports()

    # 列出 USB 设备
    usb_devices = []
    for port in ports:
        # 检查是否是 USB 设备
        if "USB" in port.hwid:
            usb_devices.append((port.device, port.description))

    return usb_devices


if __name__ == "__main__":
    devices = list_usb_devices()
    if devices:
        print("USB 设备列表:")
        for device in devices:
            print(f"设备: {device[0]}, 描述: {device[1]}")
    else:
        print("没有找到 USB 设备。")
