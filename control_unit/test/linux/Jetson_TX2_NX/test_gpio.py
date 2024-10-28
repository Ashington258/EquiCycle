import Jetson.GPIO as GPIO
import time

# 设置 GPIO 模式
GPIO.setmode(GPIO.BCM)

# 设置引脚（使用一个你确认支持的引脚）
GPIO.setup(19, GPIO.OUT)  # 使用引脚19

try:
    while True:
        GPIO.output(19, GPIO.HIGH)  # 设置引脚高电平
        time.sleep(1)
        GPIO.output(19, GPIO.LOW)  # 设置引脚低电平
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()  # 清理 GPIO 设置
