import Jetson.GPIO as GPIO
import threading
import time


class PWM:
    def __init__(self, pin, frequency=1000):
        self.pin = pin
        self.frequency = frequency
        self.duty_cycle = 0
        self.period = 1.0 / frequency
        self.running = False

        # 设置 GPIO 模式
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

    def start(self, duty_cycle):
        self.duty_cycle = duty_cycle
        self.running = True
        self.thread = threading.Thread(target=self._run_pwm)
        self.thread.start()

    def _run_pwm(self):
        high_time = self.period * (self.duty_cycle / 100.0)
        low_time = self.period * (1 - (self.duty_cycle / 100.0))

        while self.running:
            start_time = time.perf_counter()
            GPIO.output(self.pin, GPIO.HIGH)  # 设置引脚高电平
            while (time.perf_counter() - start_time) < high_time:
                pass  # 等待高电平时间

            GPIO.output(self.pin, GPIO.LOW)  # 设置引脚低电平
            start_time = time.perf_counter()
            while (time.perf_counter() - start_time) < low_time:
                pass  # 等待低电平时间

    def stop(self):
        self.running = False
        self.thread.join()  # 等待线程结束
        GPIO.output(self.pin, GPIO.LOW)  # 停止 PWM
        self.duty_cycle = 0

    def cleanup(self):
        GPIO.cleanup()  # 清理 GPIO 设置


# 示例使用
if __name__ == "__main__":
    pwm = PWM(pin=12, frequency=1000)  # 使用引脚12，频率1000Hz
    try:
        pwm.start(duty_cycle=80)  # 设置占空比为80%
    except KeyboardInterrupt:
        pass
    finally:
        pwm.stop()
        pwm.cleanup()
