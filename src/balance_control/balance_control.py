# control_module.py

import threading
import logging

# 初始化共享控制参数和锁
control_params = {}
control_params_lock = threading.Lock()


def control_layer(data):
    device = data.get("device")

    # 处理 CH100 数据
    if device == "ch100":
        logging.info(f"✅ Control layer processing CH100 data: {data}")

        # 提取欧拉角
        roll = data.get("roll", 0)
        pitch = data.get("pitch", 0)
        yaw = data.get("yaw", 0)
        euler_angles = {"roll": roll, "pitch": pitch, "yaw": yaw}

        # 提取加速度
        acceleration = data.get("acc", [0, 0, 0])  # 假设 'acc' 是加速度
        acc_data = {"x": acceleration[0], "y": acceleration[1], "z": acceleration[2]}

        # 提取角速度（陀螺仪数据）
        angular_velocity = data.get("gyr", [0, 0, 0])  # 假设 'gyr' 是角速度
        gyro_data = {
            "x": angular_velocity[0],
            "y": angular_velocity[1],
            "z": angular_velocity[2],
        }

        # 保存数据到控制参数，供控制算法使用
        with control_params_lock:
            control_params["euler_angles"] = euler_angles
            control_params["acceleration"] = acc_data
            control_params["angular_velocity"] = gyro_data

        logging.info(
            f"✨Extracted CH100 Data -> Euler Angles: {euler_angles}, Acceleration: {acc_data}, Angular Velocity: {gyro_data}"
        )

    # 处理 ODrive 数据
    elif device == "odrive":
        logging.info(f"✅ Control layer processing ODrive data: {data}")

        # 提取电机位置和电机速度
        feedback = data.get("feedback", "")
        if feedback:
            feedback_values = feedback.split()
            motor_position = float(feedback_values[0])  # 提取电机位置
            motor_speed = float(feedback_values[1])  # 提取电机速度
        else:
            motor_position = 0.0
            motor_speed = 0.0

        # 保存电机位置和速度到控制参数
        with control_params_lock:
            control_params["motor_position"] = motor_position
            control_params["motor_speed"] = motor_speed

        logging.info(
            f"✨Extracted ODrive Data -> Motor Position: {motor_position}, Motor Speed: {motor_speed}"
        )

    else:
        logging.warning(f"Unknown device data: {data}")
