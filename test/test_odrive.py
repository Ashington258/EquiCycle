import odrive
from odrive.enums import *
import time


def main():
    try:
        print("正在查找 ODrive 设备...")
        odrv0 = odrive.find_any(timeout=10)  # 设置超时为10秒
        print("成功连接到 ODrive 设备！")

        # 打印 ODrive 设备的基本信息
        print(
            f"ODrive 版本: {odrv0.fw_version_major}.{odrv0.fw_version_minor}.{odrv0.fw_version_revision}"
        )
        print(f"电源电压: {odrv0.vbus_voltage} V")

        # 读取并打印更多信息
        print("读取 ODrive 详细信息...")
        print(f"轴0状态: {odrv0.axis0.current_state}")
        print(f"轴1状态: {odrv0.axis1.current_state}")

        # 打印轴的错误信息（如果有）
        for axis in [odrv0.axis0, odrv0.axis1]:
            if axis.error != 0:
                print(f"轴 {axis.axis_number} 错误代码: {axis.error}")

        # 输出更多的 ODrive 设备信息
        print("ODrive 设备信息:")
        print(odrv0)

    except Exception as e:
        print("连接 ODrive 设备时发生错误:")
        print(str(e))


if __name__ == "__main__":
    main()
