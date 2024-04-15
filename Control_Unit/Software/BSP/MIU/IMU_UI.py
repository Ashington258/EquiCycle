# 创建一个UI界面，解析串口数据
# 串口数据为16进制数据，的样本如下：
# 55 AA DC 4A B1 07 00 6A 00 00 F2 7D 19 
# 55 AA DC 8A B1 07 00 6A 00 00 F2 7D D9 
# 55 AA DC CA B1 07 00 6A 00 00 F2 7D 99 
# 55 AA DC 0A B1 07 00 6A 00 00 F2 7D 59 
# 55 AA DC 4A B1 07 00 6A 00 00 F2 7D 19 

# 帧头：55 AA DC

# 需要解析的数据（大端字节序）是第7-8个字节(角度Pitch)，即：00 6A；第9-10个字节(角度Roll)，即：00 00；第11-12个字节（角度Yaw），即：F2 7D

import struct
import serial
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 定义端口和波特率
PORT = 'COM13'
BAUDRATE = 115200


# 解析数据
def parse_data(data):
    # 解析Pitch
    pitch = struct.unpack('>h', data[6:8])[0] * (360.0 / 65536.0)

    # 解析Roll
    roll = struct.unpack('>h', data[8:10])[0] * (360.0 / 65536.0)

    # 解析Yaw
    yaw = struct.unpack('>h', data[10:12])[0] * (360.0 / 65536.0)

    return pitch, roll, yaw

# 使用变量设置串口
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# 创建一个新的Tkinter窗口
root = tk.Tk()
root.title("IMU")

# 创建一个新的matplotlib图形
fig = plt.figure(num="IMU数据UI")

# 创建三个空的列表来存储数据
pitch_data = []
roll_data = []
yaw_data = []

# 创建一个动画函数
def animate(i):
    while True:
        # 读取一个字节的数据
        data = ser.read(1)
        if data == b'\x55':
            # 读取接下来的两个字节
            data += ser.read(2)
            if data == b'\x55\xAA\xDC':
                # 读取剩余的数据
                data += ser.read(11)
                if len(data) == 14:
                    pitch, roll, yaw = parse_data(data)
                    
                    # 添加数据到列表
                    pitch_data.append(pitch)
                    roll_data.append(roll)
                    yaw_data.append(yaw)

                    # 当数据的长度超过200时，把最旧的数据去掉
                    if len(pitch_data) > 200:
                        del pitch_data[0]
                    if len(roll_data) > 200:
                        del roll_data[0]
                    if len(yaw_data) > 200:
                        del yaw_data[0]
                    
                    # 清除当前的图形
                    plt.cla()
                    
                    # 绘制新的图形
                    plt.plot(pitch_data, label='Pitch')
                    plt.plot(roll_data, label='Roll')
                    plt.plot(yaw_data, label='Yaw')
                    
                    # 添加图例
                    plt.legend(loc='upper left')
                    # 设置标题
                    plt.title('IMU Data')
                    plt.tight_layout()

                    break
                else:
                    print('Invalid data:', data)
                break
        

# 创建一个新的动画
ani = animation.FuncAnimation(fig, animate, interval=10)

# 显示图形
plt.show()

# 主循环
root.mainloop()

# 关闭串口
ser.close()

