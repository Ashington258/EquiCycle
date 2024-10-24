import Jetson.GPIO as GPIO
import time

# 设置GPIO引脚
PWM_PIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PWM_PIN, GPIO.OUT)

# 创建PWM对象，频率为100Hz
pwm = GPIO.PWM(PWM_PIN, 1000)
pwm.start(0)  # 初始占空比为0%

try:
    while True:
        for duty_cycle in range(0, 101, 5):  # 从0%到100%
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.1)
        for duty_cycle in range(100, -1, -5):  # 从100%到0%
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
