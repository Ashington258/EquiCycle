import threading
import logging
import time

# Initialize shared control parameters and lock
control_params = {}
control_params_lock = threading.Lock()

# PID control parameters
TARGET_ROLL = 0.0  # Target roll angle for balance
Kp = 0.1  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.0  # Derivative gain
v_back = 1
stop_count = 0
stop_flag = 0

# Global variables for PID control
previous_error = 0.0
integral = 0.0
last_time = time.time()

# Define motor speed limits
MOTOR_SPEED_LIMIT = 30.0


class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0
        self.output_limits = output_limits
        self.last_value = 0

    def update(self, feedback_value):
        # 计算误差
        feedback_value = 0.5 * self.last_value + 0.5 * feedback_value
        error = self.setpoint - feedback_value
        # 计算积分
        self.integral += error
        # 计算微分
        derivative = error - self.previous_error
        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 应用输出限制
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
        # 更新上一次误差
        self.previous_error = error
        self.last_value = feedback_value
        return output

        # # 最内环：角速度环PID控制器
        # self.gyro_pid = PIDController(kp=2, ki=0, kd=1, output_limits=(-40, 40))
        # # 中间环：角度环PID控制器
        # self.angle_pid = PIDController(kp=6, ki=0, kd=6, output_limits=(-40, 40))
        # # 外环：速度环PID控制器 kp=-0.107
        # self.velocity_pid = PIDController(kp=-0.07, ki=0, kd=-0, output_limits=(-40, 40))


offset_angle = 5  # 改大往右倒


class CascadedPIDController:
    def __init__(self):
        # 最内环：角速度环PID控制器 kp=1.9, ki=0, kd=0/kp=2.37, ki=0, kd=0（8）
        self.gyro_pid = PIDController(
            kp=2.3,
            ki=0,
            kd=0.001,
            output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT),
        )
        # 中间环：角度环PID控制器kp=7, ki=0, kd=4/kp=5.7, ki=0, kd=3.7（8）
        self.angle_pid = PIDController(
            kp=5.2, ki=0, kd=3.8, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT)
        )
        # 外环：速度环PID控制器 kp=-0.085,ki=-0.001
        self.velocity_pid = PIDController(
            kp=-0.087,
            ki=-0.0003,
            kd=-0,
            output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT),
        )

        self.gyro_update_counter = 0
        self.angle_control_signal = 0
        self.velocity_control_signal = 0

    def update(self, current_angular_velocity, current_angle, current_velocity):
        # 更新速度环，每个周期计算一次
        if self.gyro_update_counter % 4 == 0:
            self.velocity_pid.setpoint = 0.0
            self.velocity_control_signal = self.velocity_pid.update(current_velocity)

        # 更新角度环，每个周期计算两次
        if self.gyro_update_counter % 2 == 0:
            self.angle_pid.setpoint = (
                self.velocity_control_signal
            )  # 将速度环的输出设为角度环的目标
            self.angle_control_signal = self.angle_pid.update(
                current_angle - offset_angle
            )

        # 更新角速度环，每个周期计算四次
        self.gyro_pid.setpoint = (
            self.angle_control_signal
        )  # 将角度环的输出设为角速度环的目标
        control_signal = self.gyro_pid.update(current_angular_velocity)

        if (current_angle - offset_angle) < -8 or (current_angle - offset_angle) > 8:
            control_signal = 0

        # 更新计数器
        if self.gyro_update_counter == 4:
            self.gyro_update_counter = 0
        self.gyro_update_counter += 1

        return control_signal


# def Balance_Calculate(data, odrive_instance, pid_controller, dt):
#     """
#     平衡控制函数，使用串级PID控制器从传感器数据获取角速度、角度、速度进行控制运算，
#     并将调整后的速度发送到ODrive电机。

#     :param data: 包含传感器数据的字典
#     :param odrive_instance: ODrive实例，用于发送电机控制命令
#     :param pid_controller: 串级PID控制器实例
#     :param dt: 时间步长
#     """
#     # 从传感器数据中提取角速度、角度、速度
#     with control_params_lock:
#         current_angular_velocity = control_params.get("angular_velocity", {}).get("x", 0.0)
#         current_angle = control_params.get("euler_angles", {}).get("roll", 0.0)
#         current_velocity = control_params.get("motor_speed", 0.0)

#     # 使用串级PID控制器进行计算
#     update_speed = pid_controller.update(current_angular_velocity, current_angle, current_velocity, dt)

#     # 更新控制参数中的电机速度
#     with control_params_lock:
#         control_params["motor_speed"] = update_speed

#     # 将新的速度命令发送给ODrive
#     # odrive_instance.motor_velocity(0, new_motor_speed, 0)

#     logging.info(
#         f" Balance Control -> Angular Velocity: {current_angular_velocity}, "
#         f"Angle: {current_angle}, Velocity: {current_velocity}, "
#         f"New_speed: {update_speed}"
#     )


def control_layer(data, odrive_instance):
    """
    Process incoming data and adjust motor speed based on control logic.
    """
    device = data.get("device")

    if device == "ch100":
        process_ch100_data(data)
    elif device == "odrive":
        process_odrive_data(data)
        adjust_motor_speed(odrive_instance, CascadePIDclass)
    else:
        logging.warning(f"Unknown device data: {data}")


def process_ch100_data(data):
    """
    Extract and save CH100 sensor data (Euler angles, acceleration, and angular velocity).
    """
    # logging.info(f"✅ Control layer processing CH100 data: {data}")

    # Extract Euler angles
    roll = data.get("roll", 0.0)
    pitch = data.get("pitch", 0.0)
    yaw = data.get("yaw", 0.0)
    euler_angles = {"roll": roll, "pitch": pitch, "yaw": yaw}

    # Extract angular velocity data
    angular_velocity = data.get(
        "gyr", [0.0, 0.0, 0.0]
    )  # Assuming 'gyr' contains angular velocity data
    gyro_data = {
        "x": angular_velocity[0],
        "y": angular_velocity[1],
        "z": angular_velocity[2],
    }

    # Save data to shared control parameters
    with control_params_lock:
        control_params["euler_angles"] = euler_angles
        control_params["gyro"] = gyro_data

    # logging.info(f"✨ Extracted CH100 Data -> Euler Angles: {euler_angles}")


def process_odrive_data(data):
    """
    Extract and save ODrive data (motor position and speed).
    """
    # logging.info(f"✅ Control layer processing ODrive data: {data}")

    # Extract motor position and speed
    feedback = data.get("feedback", "")
    if feedback:
        feedback_values = feedback.split()
        motor_position = float(feedback_values[0])
        motor_speed = float(feedback_values[1])
    # else:
    #     motor_position = 0.0
    #     motor_speed = 0.0

    # Save motor position and speed to shared control parameters
    with control_params_lock:
        control_params["motor_position"] = motor_position
        control_params["motor_speed"] = motor_speed

    # logging.info(
    #     f"✨ Extracted ODrive Data -> Motor Position: {motor_position}, "
    #     f"Motor Speed: {motor_speed}"
    # )


CascadePIDclass = CascadedPIDController()


def adjust_motor_speed(odrive_instance, PIDclass):
    """
    Adjust the motor speed using a PID controller based on the roll angle to maintain balance.
    """

    global stop_count, stop_flag
    # Get current roll angle and motor speed from control parameters
    with control_params_lock:
        gyro = control_params.get("gyro", {}).get("y", 0.0)
        angle = control_params.get("euler_angles", {}).get("roll", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)

    output = PIDclass.update(gyro, angle, motor_speed)

    # Update the motor speed with the PID output
    new_motor_speed = output

    # Apply clamping to restrict the motor speed within the defined limits
    new_motor_speed = clamp_speed(
        new_motor_speed, -MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT
    )

    # Send the updated motor speed to ODrive
    if (angle - offset_angle) < -8 or (angle - offset_angle) > 8:
        new_motor_speed = 0
        v_back = 0
    else:
        v_back = 0
        # FIX 暂时注释用于测试
    # odrive_instance.motor_velocity(0, new_motor_speed, 0)

    # stop_count += 1
    # if stop_count >= 2000:
    #     stop_flag = 1
    # if stop_flag == 1:
    #     odrive_instance.motor_velocity(1, v_back, 0)

    # logging.info(
    #     f"🔧 Output: {stop_count}, "
    #     f"New Motor Speed: {stop_flag}"
    # )


def clamp_speed(speed, min_speed, max_speed):
    """
    Clamps the motor speed between the specified limits.

    :param speed: The calculated motor speed
    :param min_speed: Minimum allowable speed
    :param max_speed: Maximum allowable speed
    :return: Clamped motor speed
    """
    if speed > max_speed:
        return max_speed
    elif speed < min_speed:
        return min_speed
    return speed
