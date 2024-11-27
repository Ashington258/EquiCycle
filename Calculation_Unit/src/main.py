from enum import Enum
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yolo_processor import YOLOProcessor
from video_processor import VideoProcessor
from control_stream.servo_stream import DirectionalControl
from control_stream.odrive_stream import ControlFlowSender
from apply_nms import apply_nms
from config import Config
from skimage.morphology import skeletonize
import torch

odrive_control = ControlFlowSender("192.168.2.113", 5000)
directional_control = DirectionalControl("192.168.2.113", 5001, 800, 2000)


class State(Enum):
    IDLE = 1  # 车道检测和元素检测
    STOP_AND_TURN = 2  # 停车和转向
    AVOID_OBSTACLE = 3  # 避障


def process_frame(
    frame,
    results,
    lane_class_name,
    horizontal_line_y,
    target_x,
    R,
    servo_midpoint,
    directional_control,
):
    """处理每帧，计算交点和舵机控制，并返回处理后的帧"""
    if frame is None:
        return frame

    filtered_boxes, filtered_scores, filtered_masks, filtered_classes = apply_nms(
        results
    )
    intersection_points = []

    # 绘制辅助横线
    cv2.line(
        frame,
        (0, horizontal_line_y),
        (frame.shape[1], horizontal_line_y),
        (255, 255, 0),
        2,
    )

    for i, box in enumerate(filtered_boxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = filtered_classes[i]
        score = filtered_scores[i]
        label = f"{lane_class_name[class_id]}: {score:.2f}"

        # 绘制边界框和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if filtered_masks is not None:
            mask = filtered_masks[i]
            mask_resized = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            binary_mask = mask_resized > 0
            skeleton = skeletonize(binary_mask)

            # 提取骨架点
            points = np.column_stack(np.where(skeleton > 0))

            if len(points) > 10:
                x = points[:, 1]
                y = points[:, 0]

                # 拟合多项式
                coefficients = np.polyfit(x, y, 3)
                polynomial = np.poly1d(coefficients)

                # 查找与横线的交点
                x_fit = np.linspace(x.min(), x.max(), 1000)
                y_fit = polynomial(x_fit)

                for xf, yf in zip(x_fit, y_fit):
                    if abs(yf - horizontal_line_y) < 1:
                        intersection_points.append((xf, yf))
                        cv2.circle(frame, (int(xf), int(yf)), 5, (0, 255, 0), -1)
                        break

    # 计算交点中点和舵机控制
    if len(intersection_points) == 2:
        center_x = int((intersection_points[0][0] + intersection_points[1][0]) / 2)
        center_y = int(horizontal_line_y)

        # 计算center_x与target_x的差值
        difference = center_x - target_x

        # 计算舵机角度
        theta = np.arctan(difference / R)

        # 映射角度到脉冲宽度（包含中值）
        pulse_width = int(abs((200 / 27) * np.degrees(theta)) + servo_midpoint)

        # 发送舵机控制命令
        directional_control.send_protocol_frame_udp(pulse_width)

        # 绘制中心点和调试信息
        cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Center: ({center_x}, {center_y})",
            (center_x + 10, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Diff: {difference}, Theta: {np.degrees(theta):.2f}°",
            (center_x + 10, center_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Pulse: {pulse_width}",
            (center_x + 10, center_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def process_idle(frame, *args, **kwargs):
    """处理车道检测和元素检测逻辑"""
    yolo_processor_lane = kwargs.get("yolo_processor_lane")
    yolo_processor_elements = kwargs.get("yolo_processor_elements")
    lane_class_name = kwargs.get("lane_class_name")
    elements_class_name = kwargs.get("elements_class_name")
    horizontal_line_y = kwargs.get("horizontal_line_y")
    target_x = kwargs.get("target_x")
    R = kwargs.get("R")
    servo_midpoint = kwargs.get("servo_midpoint")
    directional_control = kwargs.get("directional_control")

    results_lane = yolo_processor_lane.infer(frame)
    results_elements = yolo_processor_elements.infer(frame)

    # 处理车道检测结果
    frame = process_frame(
        frame,
        results_lane,
        lane_class_name,
        horizontal_line_y,
        target_x,
        R,
        servo_midpoint,
        directional_control,
    )

    # 初始化检测标志
    detected_target_element = False

    # 处理目标检测结果
    filtered_boxes, filtered_scores, filtered_masks, filtered_classes = apply_nms(
        results_elements
    )
    for i, box in enumerate(filtered_boxes):
        x1, y1, x2, y2 = map(int, box)
        elements_class_id = filtered_classes[i]
        label = f"{elements_class_name[elements_class_id]}: {filtered_scores[i]:.2f}"

        # 获取检测到的元素名称
        class_name = elements_class_name[elements_class_id]

        # 检查是否检测到指定的元素
        if class_name in ["zebra", "turn_sign"]:
            detected_target_element = True

        # 绘制目标框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 返回帧和检测标志
    return frame, detected_target_element


def process_stop_and_turn(frame, *args, **kwargs):
    """处理停车和转向逻辑"""
    # 注意由于并未使用多线程，进入该状态的车将会关闭循迹
    print("执行停车和转向任务")
    # 通过UDP协议发送停车信号 v 1 0
    odrive_control.motor_velocity(1, 0)
    time.sleep(9.9)  # 模拟停车
    print("完成停车，执行转向")
    # 首先回中值状态
    directional_control.send_protocol_frame_udp(Config.SERVO_MIDPOINT)
    # 然后向左打一个小角度
    directional_control.send_protocol_frame_udp(Config.SERVO_MIDPOINT - 50)
    # 车辆前进
    odrive_control.motor_velocity(1, 1)
    # 车辆前进2s，到达预计的位置
    time.sleep(2)

    return frame


def process_avoid_obstacle(frame, *args, **kwargs):
    """处理避障逻辑"""
    print("执行避障任务")

    # 速度降低准备避障
    odrive_control.motor_velocity(1, 0.5)
    odrive_control.motor_velocity(1, 0.5)
    # 持续向左打方向之后再持续向右打方向
    # 向左打方向 200 个脉冲
    for i in range(200):
        # 发送脉冲，向左打方向
        directional_control.send_protocol_frame_udp(
            Config.CONF_THRESH - (i % 2) * 2
        )  # 每次发送2个脉冲
        time.sleep(0.02)  # 等待 20 毫秒

    # 向右打方向 200 个脉冲
    for i in range(200):
        # 发送脉冲，向右打方向
        directional_control.send_protocol_frame_udp(
            Config.CONF_THRESH + (i % 2) * 2
        )  # 每次发送2个脉冲
        time.sleep(0.02)  # 等待 20 毫秒
    # 恢复行驶速度
    odrive_control.motor_velocity(1, Config.CAR_SPEED)
    print("避障完成")
    return frame


def main():
    # 初始化模块
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 第一个 YOLO 模型（车道检测）
    yolo_processor_lane = YOLOProcessor(
        Config.LANE_MODEL, Config.CONF_THRESH, Config.IMG_SIZE, device
    )

    # 第二个 YOLO 模型（目标检测）
    yolo_processor_elements = YOLOProcessor(
        Config.ELEMENTS_MODEL,
        Config.CONF_THRESH,
        Config.IMG_SIZE,
        device,
    )

    video_processor = VideoProcessor(Config.INPUT_SOURCE)

    # 配置参数
    lane_class_name = Config.LANE_CLASS_NAME
    elements_class_name = Config.ELEMENTS_CLASS_NAME
    horizontal_line_y = Config.HORIZONTAL_LINE_Y
    target_x = Config.TARGET_X
    R = Config.R
    servo_midpoint = Config.SERVO_MIDPOINT

    prev_time = time.time()
    fps_list = []

    # 状态初始化
    current_state = State.IDLE

    # 初始化检测计时器
    detection_start_time = None

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        current_time = time.time()

        if current_state == State.IDLE:
            # 执行车道检测和元素检测
            frame, detected_target_element = process_idle(
                frame,
                yolo_processor_lane=yolo_processor_lane,
                yolo_processor_elements=yolo_processor_elements,
                lane_class_name=lane_class_name,
                elements_class_name=elements_class_name,
                horizontal_line_y=horizontal_line_y,
                target_x=target_x,
                R=R,
                servo_midpoint=servo_midpoint,
                directional_control=directional_control,
            )

            # 更新检测计时器逻辑
            if detected_target_element:
                if detection_start_time is None:
                    detection_start_time = current_time
                else:
                    elapsed_time = current_time - detection_start_time
                    if elapsed_time >= 3:
                        current_state = State.STOP_AND_TURN
                        detection_start_time = None  # 重置计时器
            else:
                detection_start_time = None

            # 检测障碍物
            if detect_obstacle(frame):
                current_state = State.AVOID_OBSTACLE

        elif current_state == State.STOP_AND_TURN:
            # 执行停车和转向任务
            frame = process_stop_and_turn(frame)
            # 返回到 IDLE 状态
            current_state = State.IDLE

        elif current_state == State.AVOID_OBSTACLE:
            # 执行避障任务
            frame = process_avoid_obstacle(frame)
            # 返回到 IDLE 状态
            current_state = State.IDLE

        # 计算帧率
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        fps_list.append(fps)

        # 显示帧率
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 显示当前状态
        cv2.putText(
            frame,
            f"State: {current_state.name}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # 显示结果帧
        cv2.imshow("State Machine with YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 计算平均帧率
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"平均帧率: {avg_fps:.2f}")

    # 绘制帧率变化图
    plt.plot(fps_list)
    plt.axhline(avg_fps, color="r", linestyle="--", label=f"Average FPS:{avg_fps:.2f}")
    plt.title("FPS over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("FPS")
    plt.legend()
    plt.show()

    # 释放资源
    video_processor.release()


def detect_obstacle(frame):
    """检测障碍物的占位函数"""
    # 替换为真实障碍物检测逻辑
    return False


if __name__ == "__main__":
    main()
