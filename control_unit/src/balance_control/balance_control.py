import threading
import logging
import time

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

MOTOR_SPEED_LIMIT = 25  # 速度限幅


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
offset_angle = 2.05     # 左右零点
factor_dynamic_steer = 0.0067
v_basic = -3              # 直立之后后轮速度
Stand_time = 50000000       # 直立时间 2250 -> 30s
class CascadedPIDController:
    def __init__(self):
        self.gyro_pid = PIDController(kp=3.7, ki=0, kd=0.07,limit = 0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        self.angle_pid = PIDController(kp=3.5, ki=0.00001, kd=3,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        self.velocity_pid = PIDController(kp=-0.12, ki=-0.0009, kd=-0.03,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        # self.velocity_pid = PIDController(kp=-0, ki=-0, kd=-0,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
       
        #self.gyro_pid = PIDController(kp=3.13, ki=0, kd=0.07,limit = 0, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        #self.angle_pid = PIDController(kp=3.7, ki=0.000007, kd=3,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))
        #self.velocity_pid = PIDController(kp=-0.12, ki=-0.0009, kd=-0.03,limit = 10000, output_limits=(-MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT))0.75

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

def adjust_motor_speed(odrive_instance, PIDclass):
    
    global stop_count, stop_flag,flag_go,v_real
    global flag_Crossing_1,flag_Crossing_2,flag_Crossing_over,count_Crossing_2
    global dynamic_angle_steer
    
    with control_params_lock:
        gyro = control_params.get("gyro", {}).get("x", 0.0)
        angle = control_params.get("euler_angles", {}).get("pitch", 0.0)
        motor_speed = control_params.get("motor_speed", 0.0)

    output = PIDclass.update(gyro, angle-dynamic_angle_steer, motor_speed)

    new_motor_speed = output

    new_motor_speed = clamp_speed(new_motor_speed, -MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT)


    
    # 直立30s任务，优先级最低
    if flag_go == 0:
        stop_count += 1
    if stop_count >= Stand_time and flag_go == 0:
        flag_go = 1
        odrive_instance.motor_velocity(1, v_basic, 0)
        
    # # 斑马线元素处理，优先级居中
    # flag_Crossing_1 = Get_flag_from_visionPart()
    #     # 1,到达斑马线停车
    # if flag_Crossing_1 == 1 and flag_Crossing_over == 0:
    #     v_real = 0
    #     flag_Crossing_2 = 1
    #     # 2,停车计时10s
    # if flag_Crossing_2 == 1 and flag_Crossing_over == 0:
    #     flag_Crossing_1 = 0
    #     count_Crossing_2 += 1
    #     # 3,计时结束，继续前进
    # if count_Crossing_2 >= 800:
    #     count_Crossing_2 = 0
    #     flag_Crossing_2 = 0
    #     flag_Crossing_over = 1   # 该标志位保证只进行一次斑马线元素
    # if flag_Crossing_over == 1:
    #     v_real = -1
    # 倒地判断优先级最高，放在最后面
    if (angle - offset_angle) < -7 or (angle - offset_angle) > 7:
        new_motor_speed = 0
        odrive_instance.motor_velocity(1, 0, 0)
    else:
        if v_vision != None:
            odrive_instance.motor_velocity(1, v_vision, 0)
    # else:
    #     v_back = v_real
    # 发送速度
    odrive_instance.motor_velocity(0, new_motor_speed, 0)
    # odrive_instance.motor_velocity(1, v_back, 0)


def clamp_speed(speed, min_speed, max_speed):
    if speed > max_speed:
        return max_speed
    elif speed < min_speed:
        return min_speed
    return speed
