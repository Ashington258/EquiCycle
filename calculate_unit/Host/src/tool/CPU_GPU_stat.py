import curses
import psutil
import subprocess
import time
import threading
import platform

# 仅在 Windows 上导入 wmi
if platform.system() == "Windows":
    import wmi


def get_cpu_usage():
    return psutil.cpu_percent(interval=None, percpu=True)


def get_memory_usage():
    mem = psutil.virtual_memory()
    return mem.percent


def get_cpu_temperatures():
    system = platform.system()
    if system == "Windows":
        return get_cpu_temperatures_windows()
    else:
        return get_cpu_temperatures_unix()


def get_cpu_temperatures_windows():
    try:
        w = wmi.WMI(namespace="root\OpenHardwareMonitor")
        temperature_info = []
        sensors = w.Sensor()
        for sensor in sensors:
            if sensor.SensorType == "Temperature" and "CPU" in sensor.Name:
                temperature_info.append(sensor.Value)
        if temperature_info:
            return temperature_info
        else:
            return ["N/A"]
    except Exception as e:
        return [f"Error: {e}"]


def get_cpu_temperatures_unix():
    temps = psutil.sensors_temperatures()
    if not temps:
        return ["N/A"]
    # 取决于系统，可能需要调整键名
    for name, entries in temps.items():
        return [entry.current for entry in entries]
    return ["N/A"]


def get_gpu_usage():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        gpu_usage = result.stdout.strip().split("\n")
        parsed = [tuple(map(int, gpu.split(","))) for gpu in gpu_usage]
        return parsed
    except Exception as e:
        return [("N/A", "N/A", "N/A")]


def get_cpu_threads_usage():
    return psutil.cpu_percent(interval=None, percpu=True)


class SystemMonitor:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.current_screen = (
            0  # 0: CPU Usage, 1: CPU Threads, 2: CPU Temp, 3: GPU Usage
        )
        self.max_screen = 3
        self.lock = threading.Lock()
        self.cpu_usage = []
        self.memory_usage = 0
        self.cpu_temps = []
        self.gpu_usage = []
        self.cpu_threads = []
        self.running = True
        self.update_data_thread = threading.Thread(target=self.update_data, daemon=True)
        self.update_data_thread.start()

    def update_data(self):
        while self.running:
            with self.lock:
                self.cpu_usage = get_cpu_usage()
                self.memory_usage = get_memory_usage()
                self.cpu_temps = get_cpu_temperatures()
                self.gpu_usage = get_gpu_usage()
                self.cpu_threads = get_cpu_threads_usage()
            time.sleep(1)

    def draw(self):
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        title = "系统监控工具 - 使用左右箭头键切换界面，按 'q' 退出"
        self.stdscr.addstr(0, 0, title, curses.A_BOLD)

        with self.lock:
            if self.current_screen == 0:
                self.draw_cpu_usage()
            elif self.current_screen == 1:
                self.draw_cpu_threads()
            elif self.current_screen == 2:
                self.draw_cpu_temperatures()
            elif self.current_screen == 3:
                self.draw_gpu_usage()

            # 显示内存使用率
            mem_str = f"内存使用率: {self.memory_usage}%"
            self.stdscr.addstr(height - 2, 0, mem_str, curses.A_BOLD)

        self.stdscr.refresh()

    def draw_cpu_usage(self):
        self.stdscr.addstr(2, 0, "CPU 核心使用率:", curses.A_UNDERLINE)
        for idx, usage in enumerate(self.cpu_usage):
            self.stdscr.addstr(3 + idx, 2, f"核心 {idx}: {usage}%")

    def draw_cpu_threads(self):
        self.stdscr.addstr(2, 0, "CPU 线程使用率:", curses.A_UNDERLINE)
        for idx, usage in enumerate(self.cpu_threads):
            self.stdscr.addstr(3 + idx, 2, f"线程 {idx}: {usage}%")

    def draw_cpu_temperatures(self):
        self.stdscr.addstr(2, 0, "CPU 温度:", curses.A_UNDERLINE)
        for idx, temp in enumerate(self.cpu_temps):
            self.stdscr.addstr(3 + idx, 2, f"温度 {idx}: {temp}°C")

    def draw_gpu_usage(self):
        self.stdscr.addstr(2, 0, "GPU 使用率:", curses.A_UNDERLINE)
        for idx, (util, mem_used, mem_total) in enumerate(self.gpu_usage):
            self.stdscr.addstr(
                3 + idx,
                2,
                f"GPU {idx}: 使用率 {util}%, 显存 {mem_used}MB / {mem_total}MB",
            )

    def run(self):
        while self.running:
            self.draw()
            try:
                key = self.stdscr.getch()
            except:
                key = -1

            if key == curses.KEY_RIGHT or key == ord("l"):
                self.current_screen = (self.current_screen + 1) % (self.max_screen + 1)
            elif key == curses.KEY_LEFT or key == ord("h"):
                self.current_screen = (self.current_screen - 1) % (self.max_screen + 1)
            elif key == ord("q"):
                self.running = False
                break


def main(stdscr):
    # 初始化
    curses.curs_set(0)  # 隐藏光标
    stdscr.nodelay(True)  # 非阻塞输入
    stdscr.timeout(1000)  # 刷新间隔

    monitor = SystemMonitor(stdscr)
    monitor.run()


if __name__ == "__main__":
    # 检查是否在 Windows 上，如果是，则提示用户安装 OpenHardwareMonitor
    if platform.system() == "Windows":
        try:
            # 尝试连接 OpenHardwareMonitor
            w = wmi.WMI(namespace="root\OpenHardwareMonitor")
            # 这里假设 OpenHardwareMonitor 已经运行并提供 WMI 接口
        except Exception as e:
            print(
                "在 Windows 上获取 CPU 温度需要 OpenHardwareMonitor。请按以下步骤操作："
            )
            print(
                "1. 下载并运行 OpenHardwareMonitor：https://openhardwaremonitor.org/downloads/"
            )
            print("2. 启动 OpenHardwareMonitor 后，再次运行此脚本。")
            exit(1)

    curses.wrapper(main)
