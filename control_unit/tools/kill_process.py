import subprocess
import platform


def kill_process_using_port(port):
    if platform.system() == "Windows":
        # 获取占用端口的进程ID
        command = f"netstat -ano | findstr :{port}"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)

        if result.stdout:
            # 输出可能包含多行，逐行处理
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) > 4:  # 确保有足够的部分
                    pid = parts[-1]  # 最后一个部分是PID
                    if pid.isdigit() and pid != "0":  # 确保PID有效且不是0
                        print(f"Killing process {pid} on port {port}")
                        subprocess.run(f"taskkill /PID {pid} /F", shell=True)
                    else:
                        print(f"Invalid PID {pid} found on port {port}")
        else:
            print(f"No process found on port {port}")

    else:
        # 获取占用端口的进程ID
        command = f"lsof -ti:{port}"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)

        if result.stdout:
            # 解析输出，获取PID
            pids = result.stdout.split()
            for pid in pids:
                print(f"Killing process {pid} on port {port}")
                subprocess.run(f"kill -9 {pid}", shell=True)
        else:
            print(f"No process found on port {port}")


# 要结束的端口
ports = [5555, 5556]

for port in ports:
    kill_process_using_port(port)
