import zmq
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 设置ZeroMQ订阅端
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5555")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

# 数据存储
data_buffer = {
    "temperature": [],
    "pressure": [],
    "system_time_ms": [],
    "sync_time": [],
    "mag": [[], [], []],
    "acc": [[], [], []],
    "gyr": [[], [], []],
    "roll": [],
    "pitch": [],
    "yaw": [],
}

# 设置绘图
fig, axs = plt.subplots(4, 1, figsize=(10, 8))

# 绘制四个子图
lines = {
    "temperature": axs[0].plot([], [], label="Temperature (C)", color="r")[0],
    "pressure": axs[0].plot([], [], label="Pressure (Pa)", color="b")[0],
    "system_time_ms": axs[0].plot([], [], label="System Time (ms)", color="g")[0],
    "mag_x": axs[0].plot([], [], label="Mag X", color="c")[0],
    "mag_y": axs[0].plot([], [], label="Mag Y", color="m")[0],
    "mag_z": axs[0].plot([], [], label="Mag Z", color="y")[0],
    "acc_x": axs[1].plot([], [], label="Acc X (G)", color="r")[0],
    "acc_y": axs[1].plot([], [], label="Acc Y (G)", color="g")[0],
    "acc_z": axs[1].plot([], [], label="Acc Z (G)", color="b")[0],
    "gyr_x": axs[2].plot([], [], label="Gyr X (deg/s)", color="r")[0],
    "gyr_y": axs[2].plot([], [], label="Gyr Y (deg/s)", color="g")[0],
    "gyr_z": axs[2].plot([], [], label="Gyr Z (deg/s)", color="b")[0],
    "roll": axs[3].plot([], [], label="Roll (deg)", color="r")[0],
    "pitch": axs[3].plot([], [], label="Pitch (deg)", color="g")[0],
    "yaw": axs[3].plot([], [], label="Yaw (deg)", color="b")[0],
}

# 初始化绘图
for ax in axs:
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(-100, 100)


# 更新绘图数据
def update_plot(frame):
    try:
        message = subscriber.recv_string(flags=zmq.NOBLOCK)
        imu_data = json.loads(message)
        print(imu_data)  # 打印接收到的数据

        # 更新温度、压力等数据
        data_buffer["temperature"].append(imu_data.get("temperature", 0))
        data_buffer["pressure"].append(imu_data.get("pressure", 0))
        data_buffer["system_time_ms"].append(imu_data.get("system_time_ms", 0))
        data_buffer["sync_time"].append(imu_data.get("sync_time", 0))

        # 更新磁场数据
        mag_data = imu_data.get("mag", [0, 0, 0])
        for i in range(3):
            data_buffer["mag"][i].append(mag_data[i])

        # 更新加速度数据
        acc_data = imu_data.get("acc", [0, 0, 0])
        for i in range(3):
            data_buffer["acc"][i].append(acc_data[i])

        # 更新陀螺仪数据
        gyr_data = imu_data.get("gyr", [0, 0, 0])
        for i in range(3):
            data_buffer["gyr"][i].append(gyr_data[i])

        # 更新欧拉角数据
        data_buffer["roll"].append(imu_data.get("roll", 0))
        data_buffer["pitch"].append(imu_data.get("pitch", 0))
        data_buffer["yaw"].append(imu_data.get("yaw", 0))

        # 更新子图数据
        lines["temperature"].set_data(
            np.arange(len(data_buffer["temperature"])), data_buffer["temperature"]
        )
        lines["pressure"].set_data(
            np.arange(len(data_buffer["pressure"])), data_buffer["pressure"]
        )
        lines["system_time_ms"].set_data(
            np.arange(len(data_buffer["system_time_ms"])), data_buffer["system_time_ms"]
        )
        lines["mag_x"].set_data(
            np.arange(len(data_buffer["mag"][0])), data_buffer["mag"][0]
        )
        lines["mag_y"].set_data(
            np.arange(len(data_buffer["mag"][1])), data_buffer["mag"][1]
        )
        lines["mag_z"].set_data(
            np.arange(len(data_buffer["mag"][2])), data_buffer["mag"][2]
        )

        lines["acc_x"].set_data(
            np.arange(len(data_buffer["acc"][0])), data_buffer["acc"][0]
        )
        lines["acc_y"].set_data(
            np.arange(len(data_buffer["acc"][1])), data_buffer["acc"][1]
        )
        lines["acc_z"].set_data(
            np.arange(len(data_buffer["acc"][2])), data_buffer["acc"][2]
        )

        lines["gyr_x"].set_data(
            np.arange(len(data_buffer["gyr"][0])), data_buffer["gyr"][0]
        )
        lines["gyr_y"].set_data(
            np.arange(len(data_buffer["gyr"][1])), data_buffer["gyr"][1]
        )
        lines["gyr_z"].set_data(
            np.arange(len(data_buffer["gyr"][2])), data_buffer["gyr"][2]
        )

        lines["roll"].set_data(np.arange(len(data_buffer["roll"])), data_buffer["roll"])
        lines["pitch"].set_data(
            np.arange(len(data_buffer["pitch"])), data_buffer["pitch"]
        )
        lines["yaw"].set_data(np.arange(len(data_buffer["yaw"])), data_buffer["yaw"])

        # 调整x轴范围
        for ax in axs:
            ax.set_xlim(
                max(0, len(data_buffer["temperature"]) - 100),
                len(data_buffer["temperature"]),
            )

    except zmq.Again:
        pass  # No new data available

    return [line for line in lines.values()]


# 使用FuncAnimation动态更新图表
ani = FuncAnimation(fig, update_plot, interval=100)

# 显示图表
plt.tight_layout()
plt.show()
