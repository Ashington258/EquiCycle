import threading
import logging
import time

# Initialize shared control parameters and lock
control_params = {}
control_params_lock = threading.Lock()

# Define motor speed limits
MOTOR_SPEED_LIMIT = 30.0

# Global variables for motor control
v_back = 1
stop_count = 0
stop_flag = 0


def control_layer(data, odrive_instance):
    """
    Process incoming data and adjust motor speed based on control logic.
    """
    device = data.get("device")

    if device == "ch100":
        process_ch100_data(data)
    elif device == "odrive":
        process_odrive_data(data)
        adjust_motor_speed(odrive_instance)
    else:
        logging.warning(f"Unknown device data: {data}")


def process_ch100_data(data):
    """
    Extract and save CH100 sensor data (Euler angles, acceleration, and angular velocity).
    """
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


def process_odrive_data(data):
    """
    Extract and save ODrive data (motor position and speed).
    """
    # Extract motor position and speed
    feedback = data.get("feedback", "")
    if feedback:
        feedback_values = feedback.split()
        motor_position = float(feedback_values[0])
        motor_speed = float(feedback_values[1])

    # Save motor position and speed to shared control parameters
    with control_params_lock:
        control_params["motor_position"] = motor_position
        control_params["motor_speed"] = motor_speed


def adjust_motor_speed(odrive_instance):
    """
    Adjust the motor speed based on the roll angle to maintain balance.
    """
    global stop_count, stop_flag

    # Get current roll angle and motor speed from control parameters
    with control_params_lock:
        gyro = control_params.get("gyro", {}).get("y", 0.0)
        angle = control_params.get("euler_angles", {}).get("roll", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)

    # Placeholder for custom control logic (without PID)
    # For example, you can implement a simple control logic here based on angle
    control_signal = -angle  # Example: inverse of the roll angle

    # Apply clamping to restrict the motor speed within the defined limits
    new_motor_speed = clamp_speed(control_signal, -MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT)

    # Send the updated motor speed to ODrive
    if abs(angle) > 8:  # Stop the motor if the angle exceeds a threshold
        new_motor_speed = 0
        v_back = 0
    else:
        v_back = 0

    odrive_instance.motor_velocity(0, new_motor_speed, 0)


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
