# odrive_protocol.py
import serial


class ODriveAsciiProtocol:
    def __init__(self, port: str, baudrate: int = 460800):
        self.serial = serial.Serial(port, baudrate, timeout=1)

    def send_command(self, command: str):
        # Debug information: Show the command being sent
        print(f"Sending command: {command}")
        # Send command to the ODrive, with newline character
        self.serial.write(f"{command}\n".encode())

    def read_response(self):
        # Read the response from ODrive
        response = self.serial.readline().decode().strip()
        # Debug information: Show the response received
        print(f"Received response: {response}")
        return response

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
        command = f"v {motor} {velocity} {torque_ff}\n"
        self.send_command(command)

    def motor_current(self, motor: int, torque: float):
        command = f"c {motor} {torque}"
        self.send_command(command)

    def request_feedback(self, motor: int):
        command = f"f {motor}"
        self.send_command(command)
        return self.read_response()

    def update_watchdog(self, motor: int):
        command = f"u {motor}"
        self.send_command(command)

    def read_parameter(self, property_name: str):
        command = f"r {property_name}"
        self.send_command(command)
        return self.read_response()

    def write_parameter(self, property_name: str, value: float):
        command = f"w {property_name} {value}"
        self.send_command(command)

    def system_command(self, command_type: str):
        valid_commands = {"ss", "se", "sr", "sc"}
        if command_type not in valid_commands:
            raise ValueError(f"Invalid system command: {command_type}")
        self.send_command(command_type)

    def close(self):
        self.serial.close()
