from adafruit_servokit import ServoKit

# 初始化16通道舵机控制器
kit = ServoKit(channels=16)

# 设置舵机角度
kit.servo[0].angle = 90  # 第一个舵机
kit.servo[1].angle = 45  # 第二个舵机
