import time
import os
import sys

current_directory = os.getcwd()
print(f"当前工作目录: {current_directory}")
sys.path.append(r"F:/1.Project/2.Ongoing_Projects/EquiCycle/")
# 打印导入模块时的搜索路径
print("模块导入搜索路径:")
for path in sys.path:
    print(path)
from Calculation_Unit.src.control_stream.servo_stream import DirectionalControl


directional_control = DirectionalControl()


class Config:
    SERVO_MIDPOINT = 1500  # 假设的中点值，具体值根据实际情况调整


def send_pulses(total_pulses):
    # 计算脉冲的步长，正负脉冲的情况
    step = 2 if total_pulses > 0 else -2
    abs_pulses = abs(total_pulses)  # 取脉冲数的绝对值

    for i in range(abs_pulses):
        # 发送脉冲，调整方向
        directional_control.send_protocol_frame_udp(
            Config.SERVO_MIDPOINT - i * step
        )  # 每次发送2个脉冲
        print(i * step)
        time.sleep(0.02)  # 等待 20 毫秒


# 示例调用
send_pulses(100)
# directionalcontrol.send_pulses(1000)
