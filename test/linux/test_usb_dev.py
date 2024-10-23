import subprocess
import re


def check_usbutils():
    """检查并安装 usbutils"""
    try:
        subprocess.run(
            ["lsusb"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        print("lsusb 命令未找到，正在安装 usbutils...")
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "install", "-y", "usbutils"], check=True)


def get_usb_devices():
    """获取 USB 设备信息"""
    devices = []
    try:
        result = subprocess.run(
            ["lsusb"], check=True, stdout=subprocess.PIPE, text=True
        )
        lines = result.stdout.strip().split("\n")

        for line in lines:
            # 过滤出常见的设备类型
            if re.search(r"Input|Mouse|Keyboard|CH340|Serial", line, re.IGNORECASE):
                devices.append(line)
    except subprocess.CalledProcessError as e:
        print(f"获取 USB 设备信息失败: {e}")

    return devices


def main():
    check_usbutils()
    print("连接的 USB 设备信息（过滤后的）：")
    print("=========================")

    usb_devices = get_usb_devices()

    if not usb_devices:
        print("未找到符合条件的 USB 设备。")
    else:
        for line in usb_devices:
            print(line)

    print("=========================")
    print("USB 设备信息获取完成。")


if __name__ == "__main__":
    main()
