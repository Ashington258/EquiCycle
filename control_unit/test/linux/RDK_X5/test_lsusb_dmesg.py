import subprocess
import re

def get_usb_devices():
    """获取 USB 设备的信息"""
    try:
        lsusb_output = subprocess.check_output(['lsusb']).decode('utf-8')
        return lsusb_output.splitlines()
    except Exception as e:
        print(f"Error getting USB devices: {e}")
        return []

def get_dmesg_tty_info():
    """获取 dmesg 中的 tty 设备信息"""
    try:
        dmesg_output = subprocess.check_output(['dmesg']).decode('utf-8')
        tty_lines = [line for line in dmesg_output.splitlines() if 'tty' in line]
        return tty_lines
    except Exception as e:
        print(f"Error getting dmesg info: {e}")
        return []

def parse_dmesg_tty_info(tty_lines):
    """解析 dmesg 输出以获取设备与 tty 的对应关系"""
    device_mapping = {}
    for line in tty_lines:
        # 更新正则表达式以匹配更广泛的格式
        match = re.search(r'usb.*?idVendor=(\w+), idProduct=(\w+).*?attached to (ttyUSB\d+|ttyACM\d+)', line)
        if match:
            vendor_id = match.group(1)
            product_id = match.group(2)
            tty_device = match.group(3)
            device_mapping[tty_device] = f"{vendor_id}:{product_id}"
    return device_mapping

def main():
    # 获取 USB 设备信息
    usb_devices = get_usb_devices()
    print("USB Devices:")
    for device in usb_devices:
        print(device)

    # 获取 dmesg 中的 tty 设备信息
    tty_info = get_dmesg_tty_info()
    
    # 解析 tty 设备与 USB 设备 ID 的对应关系
    device_mapping = parse_dmesg_tty_info(tty_info)

    print("\nDevice Mapping:")
    if device_mapping:
        for tty_device, usb_id in device_mapping.items():
            print(f"{tty_device} -> {usb_id}")
    else:
        print("No device mapping found.")

if __name__ == "__main__":
    main()

