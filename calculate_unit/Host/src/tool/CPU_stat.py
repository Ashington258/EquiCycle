import psutil
import os
import time


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def display_cpu_info():
    while True:
        clear_console()

        # 获取CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count(logical=True)

        print("CPU 状态信息")
        print("=" * 30)
        print(f"逻辑 CPU 数量: {cpu_count}")
        print(f"当前 CPU 利用率: {cpu_percent}%")
        if cpu_freq:
            print(f"当前 CPU 频率: {cpu_freq.current:.2f} MHz")
            print(f"最小 CPU 频率: {cpu_freq.min:.2f} MHz")
            print(f"最大 CPU 频率: {cpu_freq.max:.2f} MHz")

        # 显示每个 CPU 核心的利用率
        print("\n每个 CPU 核心的利用率:")
        print("=" * 30)
        for i, percent in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
            print(f"核心 {i}: {percent}%")

        time.sleep(2)  # 每2秒更新一次


if __name__ == "__main__":
    display_cpu_info()
