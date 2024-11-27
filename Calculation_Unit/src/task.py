import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from config import Config
from yolo_processor import YOLOProcessor
from video_processor import VideoProcessor
from statemachine import StateMachine


def main():
    # 初始化配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.ELEMENTS_MODEL, Config.CONF_THRESH, Config.IMG_SIZE, device
    )

    video_processor = VideoProcessor(Config.INPUT_SOURCE)  # 确保传递输入源

    state_machine = StateMachine()  # 初始化状态机
    prev_time = time.time()
    fps_list = []

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        # YOLO推理
        results = yolo_processor.infer(frame)

        # 提取检测结果
        detections = []
        for result in results:
            for obj in result.boxes.data.tolist():
                conf = obj[4]  # 假设置信度在第5个元素
                class_id = int(obj[5])  # 假设类ID在第6个元素

                if class_id == 0 and conf > 0.9:  # 锥桶置信度大于 0.90
                    detections.append("cone")
                elif class_id == 1 and conf > 0.85:  # 斑马线置信度大于 0.85
                    detections.append("zebra")
                elif class_id == 2 and conf > 0.85:  # 转向标志置信度大于 0.85
                    detections.append("turn_sign")

        # 更新状态机
        state_machine.transition(detections)

        # 输出当前状态
        current_state = state_machine.get_state()
        cv2.putText(
            frame,
            f"State: {current_state}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        fps_list.append(fps)

        # 显示FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 显示结果
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 计算平均帧率
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"平均帧率: {avg_fps:.2f}")

    plt.plot(fps_list)
    plt.axhline(avg_fps, color="r", linestyle="--", label=f"Average FPS:{avg_fps:.2f}")
    plt.title("FPS over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("FPS")
    plt.legend()
    plt.show()

    video_processor.release()


if __name__ == "__main__":
    main()
