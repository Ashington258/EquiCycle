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

# Global variables for PID control
previous_error = 0.0
integral = 0.0
last_time = time.time()

# Define motor speed limits
MOTOR_SPEED_LIMIT = 30.0


def control_layer(data, odrive_instance):
    """
    Process incoming data and adjust motor speed based on control logic.
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
    """
    logging.info(f"✅ Control layer processing CH100 data: {data}")

    # Extract Euler angles
    roll = data.get("roll", 0.0)
    pitch = data.get("pitch", 0.0)
    yaw = data.get("yaw", 0.0)
    euler_angles = {"roll": roll, "pitch": pitch, "yaw": yaw}

    # Save data to shared control parameters
    with control_params_lock:
        control_params["euler_angles"] = euler_angles

    logging.info(f"✨ Extracted CH100 Data -> Euler Angles: {euler_angles}")


def process_odrive_data(data):
    """
    Extract and save ODrive data (motor position and speed).
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
        f"✨ Extracted ODrive Data -> Motor Position: {motor_position}, "
        f"Motor Speed: {motor_speed}"
    )


def adjust_motor_speed(odrive_instance):
    """
    Adjust the motor speed using a PID controller based on the roll angle to maintain balance.
    """
    global previous_error, integral, last_time

    # Get current roll angle and motor speed from control parameters
    with control_params_lock:
        roll = control_params.get("euler_angles", {}).get("roll", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)

    # Calculate the current error (target - actual)
    error = TARGET_ROLL - roll

    # Get the current time to calculate the time difference
    current_time = time.time()
    delta_time = current_time - last_time if last_time else 0.0

    # Calculate the integral (accumulation of errors)
    integral += error * delta_time

    # Calculate the derivative (rate of change of error)
    derivative = (error - previous_error) / delta_time if delta_time > 0 else 0.0

    # PID output
    output = Kp * error + Ki * integral + Kd * derivative

    # Update the motor speed with the PID output
    new_motor_speed = motor_speed + output

    # Apply clamping to restrict the motor speed within the defined limits
    new_motor_speed = clamp_speed(
        new_motor_speed, -MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT
    )

    # Send the updated motor speed to ODrive
    odrive_instance.motor_velocity(0, new_motor_speed, 0)

    # Store the current error and time for the next iteration
    previous_error = error
    last_time = current_time

    logging.info(
        f"🔧 PID Control -> Error: {error}, Output: {output}, "
        f"New Motor Speed: {new_motor_speed}"
    )


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
