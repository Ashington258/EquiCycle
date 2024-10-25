import time
import os

# 选择正确的PWM芯片和通道
PWM_CHIP = "0"  # pwmchip0
PWM_CHANNEL = "0"  # pwm0
PWM_PERIOD = "1000000"  # PWM周期（微秒）
PWM_DUTY_CYCLE = "500000"  # 初始占空比（微秒）

# 导出PWM通道
with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/export", "w") as f:
    f.write(PWM_CHANNEL)

# 设置PWM周期
with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/period", "w") as f:
    f.write(PWM_PERIOD)

# 设置初始占空比
with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle", "w") as f:
    f.write(PWM_DUTY_CYCLE)

# 启动PWM
with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/enable", "w") as f:
    f.write("1")

try:
    while True:
        for duty_cycle in range(0, int(PWM_PERIOD), 50000):
            with open(
                f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle", "w"
            ) as f:
                f.write(str(duty_cycle))
            time.sleep(0.1)
        for duty_cycle in range(int(PWM_PERIOD), 0, -50000):
            with open(
                f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle", "w"
            ) as f:
                f.write(str(duty_cycle))
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    # 停止PWM
    with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/enable", "w") as f:
        f.write("0")
    # 释放PWM通道
    with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/unexport", "w") as f:
        f.write(PWM_CHANNEL)
