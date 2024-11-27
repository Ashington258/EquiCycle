import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Calculation_Unit.src.yolo_processor import YOLOProcessor
from video_processor import VideoProcessor
from directional_control import DirectionalControl
from apply_nms import apply_nms
from config import Config
from skimage.morphology import skeletonize
import torch


def process_frame(
    frame,
    results,
    class_names,
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
        label = f"{class_names[class_id]}: {score:.2f}"

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


def main():
    # 初始化模块
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(Config.INPUT_SOURCE)
    directional_control = DirectionalControl()

    # 配置参数
    class_names = Config.CLASS_NAMES
    horizontal_line_y = Config.HORIZONTAL_LINE_Y
    target_x = Config.TARGET_X
    R = Config.R
    servo_midpoint = Config.SERVO_MIDPOINT

    prev_time = time.time()
    fps_list = []

    # 等待视频流准备好
    start_time = time.time()
    while True:
        ret, frame = video_processor.read_frame()
        if ret:
            break
        if time.time() - start_time > 10:  # 如果 10 秒后仍未获取到帧，退出
            print("无法连接到视频流")
            return

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        # YOLO 推理
        results = yolo_processor.infer(frame)

        # 处理每一帧
        frame = process_frame(
            frame,
            results,
            class_names,
            horizontal_line_y,
            target_x,
            R,
            servo_midpoint,
            directional_control,
        )

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
        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", frame)
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


if __name__ == "__main__":
    main()
