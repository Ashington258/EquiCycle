import argparse
import time

import odrive


DEFAULT_AXIS = 0
DEFAULT_TIMEOUT = 10.0


def read_fet_temperature(device, axis_number):
    axis = getattr(device, f"axis{axis_number}")
    return float(axis.motor.fet_thermistor.temperature)


def format_temperature(axis_number, temperature, timestamp=None):
    timestamp = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"[{timestamp}] ODrive AXIS{axis_number} | "
        f"FET temperature={temperature:.1f} °C"
    )


def main():
    parser = argparse.ArgumentParser(
        description="只读输出 ODrive FET 热敏电阻温度，不发送运动或配置命令。"
    )
    parser.add_argument("--axis", type=int, choices=(0, 1), default=DEFAULT_AXIS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="每秒重复读取，按 Ctrl+C 停止。",
    )
    args = parser.parse_args()

    print(f"正在查找 ODrive（超时 {args.timeout:g} 秒）...")
    try:
        device = odrive.find_any(timeout=args.timeout)
    except TimeoutError as error:
        raise SystemExit(
            "未找到 ODrive。请确认 USB 已连接、设备已上电，并关闭占用设备的 odrivetool。"
        ) from error

    try:
        while True:
            try:
                temperature = read_fet_temperature(device, args.axis)
            except AttributeError as error:
                raise SystemExit(
                    f"当前固件未提供 axis{args.axis}.motor.fet_thermistor.temperature。"
                ) from error
            print(format_temperature(args.axis, temperature))
            if not args.watch:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n温度读取已停止。")


if __name__ == "__main__":
    main()
