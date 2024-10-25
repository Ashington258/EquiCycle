import Jetson.GPIO as GPIO
import time


class SoftwarePWM:
    def __init__(self, pin, frequency=1000):
        self.pin = pin
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.duty_cycle = 0  # 初始化占空比为 0%

        # 设置 GPIO 模式
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

    def start(self, duty_cycle):
        """启动 PWM 并设置占空比"""
        self.duty_cycle = duty_cycle

    def stop(self):
        """停止 PWM 并清理 GPIO 设置"""
        GPIO.cleanup()

    def change_duty_cycle(self, duty_cycle):
        """动态改变占空比"""
        self.duty_cycle = duty_cycle

    def run(self):
        """运行 PWM 模拟，控制高低电平的时间"""
        try:
            while True:
                high_time = self.duty_cycle * self.period
                low_time = (1 - self.duty_cycle) * self.period

                GPIO.output(self.pin, GPIO.HIGH)
                time.sleep(high_time)

                GPIO.output(self.pin, GPIO.LOW)
                time.sleep(low_time)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# 实例化软件 PWM 类
pwm = SoftwarePWM(pin=12, frequency=1000)  # 1000Hz 的频率
pwm.start(0.5)  # 启动并设置占空比为 50%

# 在主循环中运行 PWM
pwm.run()  # 会持续运行，直到用户中断 (Ctrl + C)
