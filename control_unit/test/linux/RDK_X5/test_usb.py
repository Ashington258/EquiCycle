import usb.core
import usb.util

def list_usb_devices():
    # 查找所有 USB 设备
    devices = usb.core.find(find_all=True)

    if not devices:
        print("没有找到 USB 设备。")
        return

    print("找到以下 USB 设备：")
    for device in devices:
        print(f"设备 ID: {device.idVendor:04x}:{device.idProduct:04x} - {usb.util.get_string(device, device.iProduct)}")

if __name__ == "__main__":
    list_usb_devices()
