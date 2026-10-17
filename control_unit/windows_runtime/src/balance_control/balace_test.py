import threading
import logging
import time

# 初始化共享控制参数和锁
control_params = {}
control_params_lock = threading.Lock()

# 定义电机速度限制
MOTOR_SPEED_LIMIT = 30.0

# 电机控制的全局变量
v_back = 1
stop_count = 0
stop_flag = 0


def control_layer(data, odrive_instance):
    """
    处理传入的数据并根据控制逻辑调整电机速度。

    :param data: 包含设备数据的字典
    :param odrive_instance: ODrive 实例，用于控制电机
    """
    device = data.get("device")

    if device == "ch100":
        process_ch100_data(data)
    elif device == "odrive":
        process_odrive_data(data)
        adjust_motor_speed(odrive_instance)
    else:
        logging.warning(f"未知设备数据: {data}")


def process_ch100_data(data):
    """
    提取并保存 CH100 传感器数据（欧拉角、加速度和角速度）。

    :param data: 包含 CH100 传感器数据的字典
    """
    # 提取欧拉角
    roll = data.get("roll", 0.0)
    pitch = data.get("pitch", 0.0)
    yaw = data.get("yaw", 0.0)
    euler_angles = {"roll": roll, "pitch": pitch, "yaw": yaw}

    # 提取角速度数据
    angular_velocity = data.get("gyr", [0.0, 0.0, 0.0])  # 假设 'gyr' 包含角速度数据
    gyro_data = {
        "x": angular_velocity[0],
        "y": angular_velocity[1],
        "z": angular_velocity[2],
    }

    # 将数据保存到共享控制参数中
    with control_params_lock:
        control_params["euler_angles"] = euler_angles
        control_params["gyro"] = gyro_data


def process_odrive_data(data):
    """
    提取并保存 ODrive 数据（电机位置和速度）。

    :param data: 包含 ODrive 数据的字典
    """
    # 提取电机位置和速度
    feedback = data.get("feedback", "")
    if feedback:
        feedback_values = feedback.split()
        motor_position = float(feedback_values[0])
        motor_speed = float(feedback_values[1])

        # 将电机位置和速度保存到共享控制参数中
        with control_params_lock:
            control_params["motor_position"] = motor_position
            control_params["motor_speed"] = motor_speed


def clamp_speed(speed, min_speed, max_speed):
    """
    将电机速度限制在指定的范围内。

    :param speed: 计算出的电机速度
    :param min_speed: 最小允许速度
    :param max_speed: 最大允许速度
    :return: 限制后的电机速度
    """
    if speed > max_speed:
        return max_speed
    elif speed < min_speed:
        return min_speed
    return speed


def adjust_motor_speed(odrive_instance):
    """
    根据滚转角度调整电机速度以保持平衡。

    :param odrive_instance: ODrive 实例，用于控制电机
    """
    global stop_count, stop_flag

    # 从控制参数中获取当前的滚转角度和电机速度
    with control_params_lock:
        gyro = control_params.get("gyro", {}).get("y", 0.0)
        angle = control_params.get("euler_angles", {}).get("roll", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)

    # 自定义控制逻辑的占位符（不使用 PID）
    control_signal = -angle  # 示例：滚转角度的反向

    # 限制电机速度在定义的范围内
    new_motor_speed = clamp_speed(control_signal, -MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT)

    # 如果角度超过阈值，则停止电机
    if abs(angle) > 8:
        new_motor_speed = 0
        global v_back
        v_back = 0
    else:
        v_back = 0

    # 将更新后的电机速度发送到 ODrive
    odrive_instance.motor_velocity(0, new_motor_speed, 0)
