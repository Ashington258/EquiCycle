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
v_back = 0
stop_count = 0
stop_flag = 0


# Global variables for PID control
previous_error = 0.0
integral = 0.0
last_time = time.time()

# Define motor speed limits
MOTOR_SPEED_LIMIT = 35


class PIDController:
    def __init__(self, kp, ki, kd, limit, setpoint=0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0
        self.output_limits = output_limits
        self.last_value = 0
        self.limit  = limit

    def update(self, feedback_value):
        feedback_value = 0.5 * self.last_value + 0.5 * feedback_value
        error = self.setpoint - feedback_value
        self.integral += error
        if self.integral > self.limit:
          self.integral = self.limit
        elif self.integral < -self.limit:
          self.integral = -self.limit
        derivative = (error - self.previous_error)
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
        
        self.previous_error = error
        self.last_value = feedback_value
        return output



# self.gyro_pid = PIDController(kp=2.3, ki=0, kd=0.001, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
 # self.angle_pid = PIDController(kp=5.2, ki=0, kd=3.8, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
# self.velocity_pid = PIDController(kp=-0.088, ki=-0.0003, kd=-0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))


#  self.gyro_pid = PIDController(kp=2.7, ki=0, kd=0.001,limit = 0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
#      self.angle_pid = PIDController(kp=4.8, ki=0.00001, kd=2.7,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
 #       self.velocity_pid = PIDController(kp=-0.093, ki=-0.0003, kd=-0,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))

# offset_angle = 1.7 + (parameters['pulse_value'] - 978)
offset_angle = 1.7
v_real = 0
Stand_time = 2250 # 5ms计数器加一，2000为10s
class CascadedPIDController:
    def __init__(self):
        self.gyro_pid = PIDController(kp=3.2, ki=0, kd=0.07,limit = 0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        self.angle_pid = PIDController(kp=4.43, ki=0.00001, kd=2.63,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        self.velocity_pid = PIDController(kp=-0.095, ki=-0.00117, kd=-0.003,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        # self.velocity_pid = PIDController(kp=-0, ki=-0, kd=-0,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
       
        # self.gyro_pid = PIDController(kp=3.0, ki=0, kd=0,limit = 0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        # self.angle_pid = PIDController(kp=4.5, ki=0.000017, kd=2.8,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        # self.velocity_pid = PIDController(kp=-0.095, ki=-0.00117, kd=-0.015,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))

        self.gyro_update_counter = 0
        self.angle_control_signal = 0
        self.velocity_control_signal = 0

    def update(self, current_angular_velocity, current_angle, current_velocity):
        if self.gyro_update_counter % 4 == 0:
            self.velocity_pid.setpoint = 0.0
            self.velocity_control_signal = self.velocity_pid.update(current_velocity)

        if self.gyro_update_counter % 2 == 0:
            self.angle_pid.setpoint = self.velocity_control_signal
            self.angle_control_signal = self.angle_pid.update(current_angle - offset_angle)

        self.gyro_pid.setpoint = self.angle_control_signal
        control_signal = self.gyro_pid.update(current_angular_velocity)

        if (current_angle - offset_angle) < -8 or (current_angle - offset_angle) > 8:
            control_signal = 0

        if self.gyro_update_counter == 4:
            self.gyro_update_counter = 0
        self.gyro_update_counter += 1

        return control_signal


def control_layer(data, odrive_instance):
    device = data.get("device")

    if device == "ch100":
        process_ch100_data(data)
    elif device == "odrive":
        process_odrive_data(data)
        adjust_motor_speed(odrive_instance, CascadePIDclass)
    else:
        logging.warning(f"Unknown device data: {data}")


def process_ch100_data(data):
    roll = data.get("roll", 0.0)
    pitch = data.get("pitch", 0.0)
    yaw = data.get("yaw", 0.0)
    euler_angles = {"roll": roll, "pitch": pitch, "yaw": yaw}

    angular_velocity = data.get("gyr", [0.0, 0.0, 0.0])
    gyro_data = {"x": angular_velocity[0], "y": angular_velocity[1], "z": angular_velocity[2]}

    with control_params_lock:
        control_params["euler_angles"] = euler_angles
        control_params["gyro"] = gyro_data


def process_odrive_data(data):
    feedback = data.get("feedback", "")
    if feedback:
        feedback_values = feedback.split()
        motor_position = float(feedback_values[0])
        motor_speed = float(feedback_values[1])

    with control_params_lock:
        control_params["motor_position"] = motor_position
        control_params["motor_speed"] = motor_speed


CascadePIDclass = CascadedPIDController()

def adjust_motor_speed(odrive_instance, PIDclass):
    global stop_count, stop_flag
    with control_params_lock:
        gyro = control_params.get("gyro", {}).get("x", 0.0)
        angle = control_params.get("euler_angles", {}).get("pitch", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)

    output = PIDclass.update(gyro, angle, motor_speed)

    new_motor_speed = output

    new_motor_speed = clamp_speed(new_motor_speed, -MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT)

    if (angle - offset_angle) < -7 or (angle - offset_angle) > 7:
        new_motor_speed = 0
        v_back = 0
    else:
        v_back = v_real
    odrive_instance.motor_velocity(0, new_motor_speed, 0)

    stop_count += 1
    if stop_count >= Stand_time:
        stop_flag = 1
    if stop_flag == 1:
        odrive_instance.motor_velocity(1, v_back, 0)


def clamp_speed(speed, min_speed, max_speed):
    if speed > max_speed:
        return max_speed
    elif speed < min_speed:
        return min_speed
    return speed
