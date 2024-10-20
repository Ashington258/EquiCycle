# control_module.py

import threading
import logging

# Initialize shared control parameters and lock
control_params = {}
control_params_lock = threading.Lock()

# Control parameters
TARGET_ROLL = 0.0  # Target roll angle for balance
CONTROL_GAIN = 1.0  # Control gain to adjust the motor response


def control_layer(data, odrive_instance):
    """
    Process incoming data and adjust motor speed based on control logic.

    :param data: Dictionary containing data from sensors or devices
    :param odrive_instance: Instance of ODrive used to send motor control commands
    """
    device = data.get("device")

    if device == "ch100":
        process_ch100_data(data)
        adjust_motor_speed(odrive_instance)
    elif device == "odrive":
        process_odrive_data(data)
    else:
        logging.warning(f"Unknown device data: {data}")


def process_ch100_data(data):
    """
    Extract and save CH100 sensor data (Euler angles, acceleration, and angular velocity).

    :param data: Dictionary containing CH100 sensor data
    """
    logging.info(f"✅ Control layer processing CH100 data: {data}")

    # Extract Euler angles
    roll = data.get("roll", 0.0)
    pitch = data.get("pitch", 0.0)
    yaw = data.get("yaw", 0.0)
    euler_angles = {"roll": roll, "pitch": pitch, "yaw": yaw}

    # Extract acceleration data
    acceleration = data.get("acc", [0.0, 0.0, 0.0])
    acc_data = {"x": acceleration[0], "y": acceleration[1], "z": acceleration[2]}

    # Extract angular velocity data (gyroscope)
    angular_velocity = data.get("gyr", [0.0, 0.0, 0.0])
    gyro_data = {
        "x": angular_velocity[0],
        "y": angular_velocity[1],
        "z": angular_velocity[2],
    }

    # Save data to shared control parameters
    with control_params_lock:
        control_params["euler_angles"] = euler_angles
        control_params["acceleration"] = acc_data
        control_params["angular_velocity"] = gyro_data

    logging.info(
        f"✨Extracted CH100 Data -> Euler Angles: {euler_angles}, "
        f"Acceleration: {acc_data}, Angular Velocity: {gyro_data}"
    )


def process_odrive_data(data):
    """
    Extract and save ODrive data (motor position and speed).

    :param data: Dictionary containing ODrive data
    """
    logging.info(f"✅ Control layer processing ODrive data: {data}")

    # Extract motor position and speed
    feedback = data.get("feedback", "")
    if feedback:
        feedback_values = feedback.split()
        motor_position = float(feedback_values[0])
        motor_speed = float(feedback_values[1])
    else:
        motor_position = 0.0
        motor_speed = 0.0

    # Save motor position and speed to shared control parameters
    with control_params_lock:
        control_params["motor_position"] = motor_position
        control_params["motor_speed"] = motor_speed

    logging.info(
        f"✨Extracted ODrive Data -> Motor Position: {motor_position}, "
        f"Motor Speed: {motor_speed}"
    )


def adjust_motor_speed(odrive_instance):
    """
    Adjust the motor speed based on the roll angle to maintain balance.

    :param odrive_instance: Instance of ODrive used to send motor control commands
    """
    # Get current roll angle and motor speed from control parameters
    with control_params_lock:
        roll = control_params.get("euler_angles", {}).get("roll", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)

    # Calculate control error and speed adjustment
    error = TARGET_ROLL - roll
    speed_adjustment = CONTROL_GAIN * error
    new_motor_speed = motor_speed + speed_adjustment

    # Update motor speed in the control parameters
    with control_params_lock:
        control_params["motor_speed"] = new_motor_speed

    # Send the updated motor speed to ODrive
    # odrive_instance.motor_velocity(0, new_motor_speed, 0)

    logging.info(
        f"🔧 Control Algorithm -> Roll Error: {error}, "
        f"Speed Adjustment: {speed_adjustment}, New Motor Speed: {new_motor_speed}"
    )
