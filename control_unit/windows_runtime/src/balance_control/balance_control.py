import threading
import logging
import time
import math

# Initialize shared control parameters and lock
control_params = {}
control_params_lock = threading.Lock()

# Global variables for PID control
previous_error = 0.0
integral = 0.0
last_time = time.time()

flag_Crossing_1 = 0
flag_Crossing_2 = 0
flag_Crossing_over = 0
count_Crossing_2 = 0

stop_count = 0          # 直立计数变量
flag_go = 0             # 直立元素完成标志

MOTOR_SPEED_LIMIT = 15.0
MOTOR_SPEED_SLEW_RATE = 4000.0
PITCH_DOCKED_SETPOINT = 9.47
PITCH_BALANCE_SETPOINT = 0.0
PITCH_STOP_ANGLE = 12.0
PITCH_STALE_TIMEOUT = 0.25
BALANCE_MODE_DOCKED = "docked"
BALANCE_MODE_ACTIVE = "balance"
balance_mode = BALANCE_MODE_DOCKED


class PIDController:
    def __init__(self, kp, ki, kd, integral_limit, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limits = output_limits
        self.setpoint = 0.0
        self.integral = 0.0
        self.previous_error = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, feedback_value, dt):
        error = self.setpoint - feedback_value
        self.integral = clamp_speed(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        derivative = 0.0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return clamp_speed(output, *self.output_limits)



# self.gyro_pid = PIDController(kp=2.3, ki=0, kd=0.001, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
 # self.angle_pid = PIDController(kp=5.2, ki=0, kd=3.8, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
# self.velocity_pid = PIDController(kp=-0.088, ki=-0.0003, kd=-0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))


#  self.gyro_pid = PIDController(kp=2.7, ki=0, kd=0.001,limit = 0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
#      self.angle_pid = PIDController(kp=4.8, ki=0.00001, kd=2.7,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
 #       self.velocity_pid = PIDController(kp=-0.093, ki=-0.0003, kd=-0,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))

# offset_angle = 1.7 + (parameters['pulse_value'] - 978)
offset_angle = 2.05     # 左右零点
factor_dynamic_steer = 0.0067
v_basic = -3              # 直立之后后轮速度
Stand_time = 50000000       # 直立时间 2250 -> 30s
class CascadedPIDController:
    def __init__(self):
        self.velocity_pid = PIDController(0.0, 0.0, 0.0, 800.0, (-8.0, 8.0))
        self.angle_pid = PIDController(0, 0.0, 0, 400.0, (-25.0, 25.0))
        self.gyro_pid = PIDController(1.59, 0.0, 0, 0.0, (-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        self.reset()

    def reset(self):
        for pid in (self.velocity_pid, self.angle_pid, self.gyro_pid):
            pid.reset()
        self.update_counter = 0
        self.velocity_dt = 0.0
        self.angle_dt = 0.0
        self.velocity_target = 0.0
        self.angle_target = 0.0

    def update(self, angular_velocity, angle, velocity, dt, angle_setpoint):
        self.velocity_dt += dt
        self.angle_dt += dt
        if self.update_counter % 4 == 0:
            self.velocity_pid.setpoint = 0.0
            self.velocity_target = self.velocity_pid.update(velocity, self.velocity_dt)
            self.velocity_dt = 0.0
        if self.update_counter % 2 == 0:
            self.angle_pid.setpoint = self.velocity_target
            self.angle_target = self.angle_pid.update(
                angle - angle_setpoint, self.angle_dt
            )
            self.angle_dt = 0.0
        self.gyro_pid.setpoint = self.angle_target
        raw_output = self.gyro_pid.update(angular_velocity, dt)
        self.update_counter = (self.update_counter + 1) % 4
        return self.velocity_target, self.angle_target, raw_output


def control_layer(data, odrive_instance):
    device = data.get("device")

    if device == "ch100":
        process_ch100_data(data)
    elif device == "odrive":
        process_odrive_data(data)
        return adjust_motor_speed(odrive_instance)
    else:
        logging.warning(f"Unknown device data: {data}")
    return None


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
        control_params["pitch_received_at"] = time.monotonic()


def get_pitch_status():
    with control_params_lock:
        pitch = control_params.get("euler_angles", {}).get("pitch", 0.0)
        received_at = control_params.get("pitch_received_at", 0.0)
    is_fresh = time.monotonic() - received_at <= PITCH_STALE_TIMEOUT
    return pitch, is_fresh


def set_balance_mode(mode):
    global balance_mode
    if mode not in (BALANCE_MODE_DOCKED, BALANCE_MODE_ACTIVE):
        raise ValueError(f"Unknown balance mode: {mode!r}")
    with control_params_lock:
        balance_mode = mode
        control_params["last_control_at"] = None
    CascadePIDclass.reset()


def get_balance_status():
    with control_params_lock:
        mode = balance_mode
    setpoint = (
        PITCH_BALANCE_SETPOINT
        if mode == BALANCE_MODE_ACTIVE
        else PITCH_DOCKED_SETPOINT
    )
    return mode, setpoint


def get_tuning_snapshot():
    with control_params_lock:
        return {
            "pitch_deg": control_params.get("euler_angles", {}).get("pitch", 0.0),
            "gyro_x_deg_s": control_params.get("gyro", {}).get("x", 0.0),
            "motor_speed_rev_s": control_params.get("motor_speed", 0.0),
            "velocity_loop_output": control_params.get("velocity_loop_output", 0.0),
            "angle_loop_output": control_params.get("angle_loop_output", 0.0),
            "raw_cascade_output_rev_s": control_params.get(
                "raw_cascade_output_rev_s", 0.0
            ),
            "limited_output_rev_s": control_params.get("limited_output_rev_s", 0.0),
        }


def process_odrive_data(data):
    feedback = data.get("feedback", "")
    feedback_values = feedback.split()
    if len(feedback_values) < 2:
        raise ValueError(f"Invalid ODrive feedback: {feedback!r}")
    motor_position = float(feedback_values[0])
    motor_speed = float(feedback_values[1])

    with control_params_lock:
        control_params["motor_position"] = motor_position
        control_params["motor_speed"] = motor_speed

v_vision = None
def process_speedBack_message(message):
    # 解析并处理控制消息
    global v_vision
    parts = message.split()
    if len(parts) == 3 and parts[0] == 'v' and parts[1] == '1':
        v_vision = float(parts[2])
        
        # 你可以在这里添加更多的逻辑来处理控制指令
    else:
        v_vision = None

dynamic_angle_steer = 0.0
def process_steer_dynamicAngle(message):
    # 解析并处理控制消息
    global dynamic_angle_steer,factor_dynamic_steer
    
    if message <830:
        message = 830
    elif message > 1130:
        message = 1130
    # message = message < 830 ? 830 : ( message > 1130 ? 1130 : message )
    dynamic_angle_steer = factor_dynamic_steer * (message-978)
                
CascadePIDclass = CascadedPIDController()


def get_pid_parameters():
    """Return the configured gains and limits for all three cascade loops."""
    return {
        "velocity": {
            "kp": CascadePIDclass.velocity_pid.kp,
            "ki": CascadePIDclass.velocity_pid.ki,
            "kd": CascadePIDclass.velocity_pid.kd,
            "integral_limit": CascadePIDclass.velocity_pid.integral_limit,
            "output_limits": CascadePIDclass.velocity_pid.output_limits,
        },
        "angle": {
            "kp": CascadePIDclass.angle_pid.kp,
            "ki": CascadePIDclass.angle_pid.ki,
            "kd": CascadePIDclass.angle_pid.kd,
            "integral_limit": CascadePIDclass.angle_pid.integral_limit,
            "output_limits": CascadePIDclass.angle_pid.output_limits,
        },
        "gyro": {
            "kp": CascadePIDclass.gyro_pid.kp,
            "ki": CascadePIDclass.gyro_pid.ki,
            "kd": CascadePIDclass.gyro_pid.kd,
            "integral_limit": CascadePIDclass.gyro_pid.integral_limit,
            "output_limits": CascadePIDclass.gyro_pid.output_limits,
        },
    }


def store_control_outputs(velocity_output, angle_output, raw_output, limited_output):
    with control_params_lock:
        control_params["velocity_loop_output"] = velocity_output
        control_params["angle_loop_output"] = angle_output
        control_params["raw_cascade_output_rev_s"] = raw_output
        control_params["limited_output_rev_s"] = limited_output


def adjust_motor_speed(odrive_instance):
    now = time.monotonic()
    mode, setpoint = get_balance_status()
    if mode == BALANCE_MODE_DOCKED:
        CascadePIDclass.reset()
        store_control_outputs(0.0, 0.0, 0.0, 0.0)
        odrive_instance.motor_velocity(0, 0, 0)
        return 0.0
    with control_params_lock:
        pitch = control_params.get("euler_angles", {}).get("pitch", 0.0)
        pitch_received_at = control_params.get("pitch_received_at", 0.0)
        gyro = control_params.get("gyro", {}).get("x", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)
        last_control_at = control_params.get("last_control_at")

    values = (pitch, gyro, motor_speed)
    if not all(math.isfinite(value) for value in values):
        store_control_outputs(0.0, 0.0, 0.0, 0.0)
        odrive_instance.motor_velocity(0, 0, 0)
        raise ValueError(f"Balance inputs must be finite, got {values!r}")
    if now - pitch_received_at > PITCH_STALE_TIMEOUT:
        CascadePIDclass.reset()
        store_control_outputs(0.0, 0.0, 0.0, 0.0)
        odrive_instance.motor_velocity(0, 0, 0)
        return 0.0

    error = pitch - setpoint
    if abs(error) > PITCH_STOP_ANGLE:
        CascadePIDclass.reset()
        store_control_outputs(0.0, 0.0, 0.0, 0.0)
        odrive_instance.motor_velocity(0, 0, 0)
        return 0.0

    if last_control_at is None:
        dt = 0.02
    else:
        dt = clamp_speed(now - last_control_at, 0.001, 0.1)
    with control_params_lock:
        control_params["last_control_at"] = now
    velocity_output, angle_output, raw_output = CascadePIDclass.update(
        gyro,
        pitch,
        motor_speed,
        dt,
        setpoint,
    )
    max_speed_step = MOTOR_SPEED_SLEW_RATE * dt
    new_motor_speed = clamp_speed(
        raw_output,
        motor_speed - max_speed_step,
        motor_speed + max_speed_step,
    )
    new_motor_speed = clamp_speed(
        new_motor_speed,
        -MOTOR_SPEED_LIMIT,
        MOTOR_SPEED_LIMIT,
    )
    store_control_outputs(
        velocity_output,
        angle_output,
        raw_output,
        new_motor_speed,
    )
    odrive_instance.motor_velocity(0, new_motor_speed, 0)
    return new_motor_speed


def clamp_speed(speed, min_speed, max_speed):
    if speed > max_speed:
        return max_speed
    elif speed < min_speed:
        return min_speed
    return speed
