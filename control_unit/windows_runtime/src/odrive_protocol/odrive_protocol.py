import math
import threading

import serial
import serial.tools.list_ports


class ODriveAsciiProtocol:
    def __init__(self, port: str, baudrate: int = 460800):
        self.serial = serial.Serial(
            port,
            baudrate,
            timeout=0.25,
            write_timeout=1.0,
        )
        self.serial_lock = threading.RLock()

    def _write_locked(self, command: str):
        self.serial.write(f"{command}\n".encode("ascii"))

    def _query(self, command: str, validator=None, attempts: int = 2):
        last_error = None
        with self.serial_lock:
            for _attempt in range(attempts):
                try:
                    self.serial.reset_input_buffer()
                    self._write_locked(command)
                    response = self.serial.readline().decode("ascii").strip()
                    if not response:
                        raise TimeoutError(f"ODrive 查询无响应: {command}")
                    if validator is not None:
                        validator(response)
                    return response
                except (
                    serial.SerialTimeoutException,
                    TimeoutError,
                    UnicodeDecodeError,
                    ValueError,
                ) as exc:
                    last_error = exc
            raise TimeoutError(
                f"ODrive 查询连续 {attempts} 次失败: {command}; {last_error}"
            ) from last_error

    def send_command(self, command: str):
        with self.serial_lock:
            self._write_locked(command)

    def read_response(self):
        with self.serial_lock:
            return self.serial.readline().decode("ascii").strip()

    def motor_trajectory(self, motor: int, destination: float):
        command = f"t {motor} {destination}"
        self.send_command(command)

    def motor_position(
        self,
        motor: int,
        position: float,
        velocity_lim: float = None,
        torque_lim: float = None,
    ):
        command = f"q {motor} {position}"
        if velocity_lim is not None:
            command += f" {velocity_lim}"
        if torque_lim is not None:
            command += f" {torque_lim}"
        self.send_command(command)

    def motor_position_stream(
        self,
        motor: int,
        position: float,
        velocity_ff: float = 0.0,
        torque_ff: float = 0.0,
    ):
        command = f"p {motor} {position} {velocity_ff} {torque_ff}"
        self.send_command(command)

    def motor_velocity(self, motor: int, velocity: float, torque_ff: float = 0.0):
        command = f"v {motor} {velocity} {torque_ff}"
        self.send_command(command)

    def motor_current(self, motor: int, torque: float):
        command = f"c {motor} {torque}"
        self.send_command(command)

    def request_feedback(self, motor: int):
        def validate_feedback(response):
            values = response.split()
            if len(values) != 2 or not all(
                math.isfinite(float(value)) for value in values
            ):
                raise ValueError(f"无效反馈: {response!r}")

        return self._query(f"f {motor}", validator=validate_feedback)

    def update_watchdog(self, motor: int):
        command = f"u {motor}"
        self.send_command(command)

    def read_parameter(self, property_name: str):
        return self._query(f"r {property_name}")

    def write_parameter(self, property_name: str, value: float):
        command = f"w {property_name} {value}"
        self.send_command(command)

    def system_command(self, command_type: str):
        valid_commands = {"ss", "se", "sr", "sc"}
        if command_type not in valid_commands:
            raise ValueError(f"Invalid system command: {command_type}")
        self.send_command(command_type)

    def check_info(self):
        # 发送一系列指令以获取 ODrive 的状态和参数
        print("请求 ODrive 设备的所有信息...")
        responses = {}

        # 查询固件版本
        responses["firmware_version"] = self.read_parameter("fw_version")
        # 查询电源电压
        responses["vbus_voltage"] = self.read_parameter("vbus_voltage")
        # 查询轴状态
        responses["axis0_state"] = self.request_feedback(0)
        responses["axis1_state"] = self.request_feedback(1)

        # 可以根据需要添加更多的查询指令
        # 例如：查询轴的错误状态
        responses["axis0_error"] = self.read_parameter("axis0.error")
        responses["axis1_error"] = self.read_parameter("axis1.error")

        print("获取的信息:", responses)
        return responses

    def close(self):
        with self.serial_lock:
            self.serial.close()
            return not self.serial.is_open

    @staticmethod
    def find_odrive():
        # 常见的波特率列表，按常见性排序
        common_baudrates = [115200, 9600, 230400, 57600, 500000, 1000000]

        # 列出所有可用的串口
        available_ports = list(serial.tools.list_ports.comports())
        print("正在扫描可用的串口...")

        # 尝试每个串口和波特率
        for port_info in available_ports:
            port = port_info.device
            print(f"尝试连接端口: {port}")
            for baudrate in common_baudrates:
                print(f"尝试波特率: {baudrate}")
                try:
                    # 打开串口
                    ser = serial.Serial(port, baudrate, timeout=1)
                    ser.write(b"r vbus_voltage\n")  # 发送简单的命令，检查是否有响应
                    response = ser.readline().decode().strip()
                    if response:  # 如果有响应，说明可能是 ODrive 设备
                        print(f"找到 ODrive 设备, 端口: {port}, 波特率: {baudrate}")
                        ser.close()
                        return ODriveAsciiProtocol(port, baudrate)
                    ser.close()
                except (serial.SerialException, OSError) as e:
                    # 如果尝试失败，继续尝试下一个组合
                    print(f"连接失败: {e}")

        print("未找到 ODrive 设备。")
        return None
