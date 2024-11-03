import subprocess
import time
import os


def get_gpu_info():
    # 调用 nvidia-smi 命令
    try:
        output = subprocess.check_output(
            ["nvidia-smi"], encoding="utf-8", errors="ignore"
        )
        return output
    except Exception as e:
        return str(e)


def clear_console():
    # 清屏
    os.system("cls" if os.name == "nt" else "clear")


def main():
    while True:
        clear_console()
        gpu_info = get_gpu_info()
        print(gpu_info)
        time.sleep(0.1)  # 每2秒更新一次


if __name__ == "__main__":
    main()
