import Jetson.GPIO as GPIO
import time

# 设置引脚编号模式
GPIO.setmode(GPIO.BCM)

# 设置引脚12为输出模式
pwm_pin = 12
GPIO.setup(pwm_pin, GPIO.OUT)

# 创建PWM对象，频率为100Hz
pwm = GPIO.PWM(pwm_pin, 100)

# 启动PWM，初始占空比为50%
pwm.start(50)

try:
    while True:
        # 可以在这里添加其他逻辑，当前占空比为50%
        time.sleep(1)  # 每秒循环一次

except KeyboardInterrupt:
    pass  # 捕获Ctrl+C以退出

# 停止PWM信号
pwm.stop()

# 清理引脚设置
GPIO.cleanup()
