import logging
import sys
import serial
import json
import psutil  # 用于检测和终止进程

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 从 JSON 文件读取设备配置
def load_device_config(filename="EquiCycle/control_unit/src/config.json"):
    with open(filename, "r") as f:
        return json.load(f)


# 检查端口是否被占用
def is_port_in_use(port):
    """检查端口是否被占用"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False


# 释放占用端口的进程
def release_port(port):
    """释放占用端口的进程"""
    for proc in psutil.process_iter(["pid", "name"]):
        # 过滤掉系统进程
        if proc.info["pid"] == 0:
            continue
        for conn in proc.connections(kind="inet"):
            if conn.laddr.port == port:
                logging.info(
                    f"正在终止占用端口 {port} 的进程 {proc.info['name']} (PID: {proc.info['pid']})"
                )
                try:
                    proc.terminate()
                    proc.wait(timeout=3)  # 等待进程终止
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logging.error(
                        f"无法终止进程 {proc.info['name']} (PID: {proc.info['pid']})"
                    )
                return True
    return False


# 端口检测与释放
def check_and_release_ports(port_list):
    for port in port_list:
        if is_port_in_use(port):
            logging.warning(f"端口 {port} 已被占用，尝试释放...")
            if release_port(port):
                logging.info(f"端口 {port} 已成功释放")
            else:
                logging.error(f"端口 {port} 无法释放，请手动检查")
                sys.exit(1)
        else:
            logging.info(f"端口 {port} 可用")


# 设备自检
def self_check(*device_names):
    # 读取设备配置
    device_config = load_device_config()

    # 检查指定的设备端口是否正常
    zmq_ports = []
    for device_name in device_names:
        if device_name not in device_config:
            logging.error(f"未知设备名称: {device_name}")
            sys.exit(1)

        port = device_config[device_name]["port"]
        baudrate = device_config[device_name]["baudrate"]
        zmq_ports.append(device_config[device_name]["zmq_port"])  # 收集 ZMQ 端口

        try:
            # 检查设备端口
            with serial.Serial(port=port, baudrate=baudrate, timeout=1) as ser:
                logging.info(
                    f"{device_name.upper()} 自检通过: 端口连接正常，波特率正常"
                )

        except serial.SerialException as e:
            logging.error(f"{device_name.upper()} 自检失败: {e}")
            sys.exit(1)

    # 检查 ZMQ 端口是否被占用
    check_and_release_ports(zmq_ports)


# 示例调用
if __name__ == "__main__":
    self_check("ch100", "odrive")  # 同时检查 ch100 和 odrive
