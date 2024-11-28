from enum import Enum
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yolo_processor import YOLOProcessor
from video_processor import VideoProcessor
from control_stream.servo_stream import DirectionalControl
from apply_nms import apply_nms
from config import Config
from skimage.morphology import skeletonize
import torch


class State(Enum):
    IDLE = 1  # 车道检测和元素检测
    STOP_AND_TURN = 2  # 停车和转向
    AVOID_OBSTACLE = 3  # 避障


def process_frame(
    frame,
    results,
    # UPDATE 更新标签名称
    lane_class_name,
    horizontal_line_y,
    target_x,
    R,
    servo_midpoint,
    directional_control,
):
    """处理每帧，计算交点和舵机控制，并返回处理后的帧"""
    if frame is None:  # 检查帧是否为空
        return frame

    """处理每帧，计算交点和舵机控制，并返回处理后的帧"""
    filtered_boxes, filtered_scores, filtered_masks, filtered_classes = apply_nms(
        results
    )
    intersection_points = []  # 存储交点

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

            if len(points) > 10:  # 确保点数足够进行拟合
                x = points[:, 1]
                y = points[:, 0]

                # 拟合多项式
                coefficients = np.polyfit(x, y, 3)
                polynomial = np.poly1d(coefficients)

                # 查找与横线的交点
                x_fit = np.linspace(x.min(), x.max(), 1000)
                y_fit = polynomial(x_fit)

                for xf, yf in zip(x_fit, y_fit):
                    if abs(yf - horizontal_line_y) < 1:  # 找到接近横线的点
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

    # 处理目标检测结果
    filtered_boxes, filtered_scores, filtered_masks, filtered_classes = apply_nms(
        results_elements
    )
    for i, box in enumerate(filtered_boxes):
        x1, y1, x2, y2 = map(int, box)
        elements_class_id = filtered_classes[i]
        label = f"{elements_class_name[elements_class_id]}: {filtered_scores[i]:.2f}"

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
    return frame


def process_stop_and_turn(frame, *args, **kwargs):
    """处理停车和转向逻辑"""
    print("执行停车和转向任务")
    # 假设我们检测到停车线，模拟停车和转向
    time.sleep(2)  # 模拟停车
    print("完成停车，执行转向")
    return frame


def process_avoid_obstacle(frame, *args, **kwargs):
    """处理避障逻辑"""
    print("执行避障任务")
    # 模拟避障任务逻辑
    time.sleep(1)
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
    directional_control = DirectionalControl()

    # 配置参数
    # update 修改标签为labe

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

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        if current_state == State.IDLE:
            # 执行车道检测和元素检测
            frame = process_idle(
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
            # 假设触发条件为检测到停车线
            if detect_stop_line(frame):
                current_state = State.STOP_AND_TURN
            elif detect_obstacle(frame):
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
        current_time = time.time()
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


def detect_stop_line(frame):
    """检测停车线的占位函数"""
    # 替换为真实停车线检测逻辑
    return False


def detect_obstacle(frame):
    """检测障碍物的占位函数"""
    # 替换为真实障碍物检测逻辑
    return False


if __name__ == "__main__":
    main()
