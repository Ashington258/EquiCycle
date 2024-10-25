import time
import os

# 定义 PWM 通道
PWM_CHIP = '/sys/class/pwm/pwmchip0'
PWM_CHANNEL = 'pwm0'

# 导出 PWM 通道
def export_pwm():
    with open(os.path.join(PWM_CHIP, 'export'), 'w') as f:
        f.write('0')  # 假设使用 pwm0

# 设置 PWM 周期
def set_period(period_ns):
    with open(os.path.join(PWM_CHIP, PWM_CHANNEL, 'period'), 'w') as f:
        f.write(str(period_ns))

# 设置占空比
def set_duty_cycle(duty_cycle_ns):
    with open(os.path.join(PWM_CHIP, PWM_CHANNEL, 'duty_cycle'), 'w') as f:
        f.write(str(duty_cycle_ns))

# 启用 PWM
def enable_pwm():
    with open(os.path.join(PWM_CHIP, PWM_CHANNEL, 'enable'), 'w') as f:
        f.write('1')

# 禁用 PWM
def disable_pwm():
    with open(os.path.join(PWM_CHIP, PWM_CHANNEL, 'enable'), 'w') as f:
        f.write('0')

# 主程序
if __name__ == "__main__":
    export_pwm()  # 导出 PWM 通道
    time.sleep(0.1)  # 等待导出完成

    period_ns = 1000000  # 设置周期为 1 秒
    duty_cycle_ns = 500000  # 设置占空比为 50%

    set_period(period_ns)  # 设置周期
    set_duty_cycle(duty_cycle_ns)  # 设置占空比
    enable_pwm()  # 启用 PWM

    try:
        while True:
            time.sleep(1)  # 保持 PWM 输出
    except KeyboardInterrupt:
        pass
    finally:
        disable_pwm()  # 禁用 PWM
