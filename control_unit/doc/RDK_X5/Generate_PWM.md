在Ubuntu上产生PWM波形通常涉及到几个步骤，包括设置GPIO引脚和使用合适的工具或库来生成PWM信号。以下是一个基本的步骤指南：

### 1. 确认硬件支持
确保你的硬件（如树莓派、BeagleBone等）支持PWM输出，并且已经正确连接。

### 2. 安装必要的软件
你可能需要安装一些工具和库，例如`WiringPi`（适用于树莓派）或`RPi.GPIO`（Python库）。

```bash
# 对于树莓派，安装WiringPi
sudo apt-get install wiringpi

# 对于Python，安装RPi.GPIO
sudo apt-get install python3-rpi.gpio
```

### 3. 设置GPIO引脚
使用以下命令来设置GPIO引脚为PWM模式。这里以树莓派为例：

```bash
# 进入GPIO目录
cd /sys/class/pwm/pwmchip0/

# 导出PWM通道（例如通道0）
echo 0 > export

# 设置频率（例如1000000Hz）
echo 1000000 > pwm0/period

# 设置占空比（例如50%）
echo 500000 > pwm0/duty_cycle

# 启用PWM
echo 1 > pwm0/enabled
```

### 4. 使用Python代码生成PWM
如果你选择使用Python，可以使用以下代码示例：

```python
import RPi.GPIO as GPIO
import time

# 设置GPIO模式
GPIO.setmode(GPIO.BCM)

# 设置PWM引脚
pwm_pin = 18  # 这里假设使用GPIO 18
GPIO.setup(pwm_pin, GPIO.OUT)

# 创建PWM实例，频率为1000Hz
pwm = GPIO.PWM(pwm_pin, 1000)

# 启动PWM，初始占空比为0%
pwm.start(0)

try:
    while True:
        for duty_cycle in range(0, 101, 5):  # 从0%到100%
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.1)  # 等待0.1秒
        for duty_cycle in range(100, -1, -5):  # 从100%到0%
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.1)  # 等待0.1秒
except KeyboardInterrupt:
    pass
finally:
    pwm.stop()  # 停止PWM
    GPIO.cleanup()  # 清理GPIO设置
```

### 5. 运行代码
保存代码为`pwm_test.py`，然后在终端中运行：

```bash
python3 pwm_test.py
```

### 6. 调试
如果PWM信号没有正常工作，可以检查以下几点：
- 确保GPIO引脚连接正确。
- 确保你有足够的权限访问GPIO（可能需要以root用户运行）。
- 检查是否有其他程序正在使用相同的GPIO引脚。

以上步骤应能帮助你在Ubuntu上成功产生PWM波形。如果你使用的是其他硬件平台，步骤可能会有所不同，请参考相应的文档。