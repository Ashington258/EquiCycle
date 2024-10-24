import Jetson.GPIO as GPIO
import time

class Servo:
    def __init__(self, pin, max_angle=180):
        self.pin = pin
        self.frequency = 50  # 舵机的标准频率为 50Hz
        self.period = 1.0 / self.frequency  # 20ms 的周期
        self.min_pulse = 0.0005  # 500us -> 0.5ms
        self.max_pulse = 0.0025  # 2500us -> 2.5ms
        self.max_angle = max_angle

        # 设置 GPIO 模式
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

        # 创建 PWM 实例
        self.pwm = GPIO.PWM(self.pin, self.frequency)
        self.pwm.start(0)  # 初始化为 0% 占空比

    def angle_to_pulse(self, angle):
        """将角度映射到脉宽范围"""
        pulse_range = self.max_pulse - self.min_pulse
        pulse = self.min_pulse + (pulse_range * angle / self.max_angle)
        return pulse

    def set_angle(self, angle):
        """根据给定角度，控制舵机"""
        pulse = self.angle_to_pulse(angle)
        duty_cycle = pulse / self.period * 100  # 转换为占空比
        self.pwm.ChangeDutyCycle(duty_cycle)  # 更新 PWM 占空比

    def cleanup(self):
        """清理 GPIO 设置"""
        self.pwm.stop()  # 停止 PWM
        GPIO.cleanup()

# 实例化舵机类
servo = Servo(pin=12, max_angle=180)

try:
    while True:
        # 设置角度（例如从 0 度到 180 度，再回到 0 度）
        for angle in range(0, 181, 10):  # 0 度到 180 度，步长为 10
            servo.set_angle(angle)
            time.sleep(0.5)  # 设置角度后等待 0.5 秒

        for angle in range(180, -1, -10):  # 180 度到 0 度，步长为 10
            servo.set_angle(angle)
            time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    servo.cleanup()  # 清理 GPIO 设置
