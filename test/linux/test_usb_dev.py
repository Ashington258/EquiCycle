import usb.core
import usb.util


def list_usb_devices():
    # 查找所有USB设备
    devices = usb.core.find(find_all=True)

    # 遍历并打印每个设备的信息
    for device in devices:
        try:
            print(f"Device: {device}")
            print(f"  ID: {device.idVendor}:{device.idProduct}")
            print(
                f"  Manufacturer: {usb.util.get_string(device, device.iManufacturer)}"
            )
            print(f"  Product: {usb.util.get_string(device, device.iProduct)}")
            print(
                f"  Serial Number: {usb.util.get_string(device, device.iSerialNumber)}"
            )
            print(f"  Bus Number: {device.bus}")
            print(f"  Device Address: {device.address}")
            print("-" * 40)
        except Exception as e:
            print(f"Error retrieving info for device: {e}")


if __name__ == "__main__":
    list_usb_devices()
