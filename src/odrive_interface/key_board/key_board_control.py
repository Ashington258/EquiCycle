import serial
import keyboard
import time

# 配置串口
odrive_port = "COM10"  # 根据你的系统修改端口
baud_rate = 460800

# 打开串口
odrive = serial.Serial(odrive_port, baud_rate, timeout=1)  # 增加timeout来避免阻塞

# 初始化速度
current_speed = 0.0
speed_increment = 0.1  # 每次按键增减的速度
max_speed = 5.0  # 最大速度


# 定义一个发送并接收返回信息的函数
def send_and_receive(command):
    # 发送指令
    odrive.write(f"{command}\n".encode())
    # 短暂等待设备响应
    time.sleep(0.1)
    # 读取返回信息
    if odrive.in_waiting > 0:
        response = odrive.readline().decode().strip()
        print("设备回传:", response)
    else:
        print("设备未返回任何数据")


try:
    print("使用上下左右键控制 ODrive:")
    print("上键: 加速, 下键: 减速, 左键: 逆时针, 右键: 顺时针")

    while True:
        if keyboard.is_pressed("up"):
            current_speed += speed_increment
            if current_speed > max_speed:
                current_speed = max_speed
            command = f"odrv0.axis0.controller.vel_setpoint = {current_speed}"
            print("发送指令:", command)
            send_and_receive(command)

        elif keyboard.is_pressed("down"):
            current_speed -= speed_increment
            if current_speed < -max_speed:
                current_speed = -max_speed
            command = f"odrv0.axis0.controller.vel_setpoint = {current_speed}"
            print("发送指令:", command)
            send_and_receive(command)

        elif keyboard.is_pressed("left"):
            command = f"odrv0.axis0.controller.vel_setpoint = {-abs(current_speed)}"
            print("发送指令:", command)
            send_and_receive(command)

        elif keyboard.is_pressed("right"):
            command = f"odrv0.axis0.controller.vel_setpoint = {abs(current_speed)}"
            print("发送指令:", command)
            send_and_receive(command)

        # 停止电机
        if not (
            keyboard.is_pressed("up")
            or keyboard.is_pressed("down")
            or keyboard.is_pressed("left")
            or keyboard.is_pressed("right")
        ):
            command = "odrv0.axis0.controller.vel_setpoint = 0"
            print("发送指令:", command)
            send_and_receive(command)

        # 读取当前电机速度
        odrive.write(b"print(odrv0.axis0.encoder.vel_estimate)\n")
        if odrive.in_waiting > 0:
            velocity = odrive.readline().decode().strip()
            print("实时速度:", velocity)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("程序结束")
finally:
    odrive.close()
