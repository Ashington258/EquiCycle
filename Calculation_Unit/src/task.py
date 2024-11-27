import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from yolo_processor import YOLOProcessor
from video_processor import VideoProcessor


class Config:
    """配置参数类"""

    MODEL_PATH = "Calculation_Unit/model/elements.pt"
    INPUT_SOURCE = "dataset/video/1280.mp4"
    CONF_THRESH = 0.65
    IMG_SIZE = 640
    ROI_TOP_LEFT_RATIO = (0, 0.35)
    ROI_BOTTOM_RIGHT_RATIO = (1, 0.95)


class Utils:
    """通用工具类"""

    @staticmethod
    def resize_frame(frame, target_width):
        height, width = frame.shape[:2]
        scale = target_width / width
        target_height = int(height * scale)
        return cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )


class VideoStream:
    """网络视频流处理类"""

    def __init__(self, url):
        self.url = url
        self.bytes_data = b""
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.read_stream, daemon=True)
        self.thread.start()

    def read_stream(self):
        response = requests.get(self.url, stream=True)
        if response.status_code != 200:
            print("无法连接到视频流")
            self.running = False
            return

        for chunk in response.iter_content(chunk_size=4096):
            self.bytes_data += chunk
            a = self.bytes_data.find(b"\xff\xd8")
            b = self.bytes_data.find(b"\xff\xd9")
            if a != -1 and b != -1:
                jpg = self.bytes_data[a : b + 2]
                self.bytes_data = self.bytes_data[b + 2 :]
                self.frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False


class StateMachine:
    """状态机类，执行任务代码"""

    def __init__(self):
        self.state = "IDLE"  # 初始状态
        self.executed_tasks = {
            "zebra": False,  # 斑马线任务是否执行过
            "turn_sign": False,  # 转向标志任务是否执行过
        }
        self.cone_count = 0  # 记录锥桶的检测次数
        self.cone_task_executed = False  # 锥桶任务是否执行
        self.cone_start_time = None  # 锥桶置信度满足条件的开始时间
        self.cone_detection_active = False  # 当前是否检测到锥桶
        self.cone_confidence_met = False  # 锥桶置信度是否已达到条件

    def transition(self, detections, confidence_threshold=0.85):
        """
        根据检测结果进行状态转换并执行任务。
        detections: List[str] - 检测的类型列表（如 ["cone", "zebra"]）
        """
        current_time = time.time()  # 当前时间戳

        # 判断是否检测到锥桶
        if "cone" in detections:
            self.handle_cone_detection(current_time)
        else:
            # 如果没有检测到锥桶，重置相关状态
            self.reset_cone_detection()

        # 处理其他任务
        for detection in detections:
            if detection == "zebra" and not self.executed_tasks["zebra"]:
                # 如果检测到斑马线且任务未执行，则优先执行斑马线任务
                self.state = "DETECTED_ZEBRA"
                self.execute_task2()
            elif detection == "turn_sign" and not self.executed_tasks["turn_sign"]:
                # 转向任务仅在斑马线任务已执行的情况下才会执行
                if self.executed_tasks["zebra"]:  # 依赖于斑马线任务完成
                    self.state = "DETECTED_TURN_SIGN"
                    self.execute_task3()
            else:
                # 如果没有未执行的任务，回到空闲状态
                self.state = "IDLE"

    def handle_cone_detection(self, current_time):
        """处理锥桶检测逻辑"""
        if not self.cone_confidence_met:
            # 如果是首次达到置信度阈值，记录开始时间
            self.cone_start_time = current_time
            self.cone_confidence_met = True
            print("Cone detection started. Waiting for 3 seconds...")
        else:
            # 检查置信度是否持续超过3秒
            if (
                current_time - self.cone_start_time >= 3
                and not self.cone_detection_active
            ):
                # 确认锥桶检测成功，并标记为“活动检测状态”
                self.cone_detection_active = True
                self.cone_count += 1  # 每次有效检测到锥桶时计数加1

                if self.cone_count < 3:
                    # 第一次和第二次检测到锥桶时，仅显示锥桶序号
                    print(
                        f"Detected cone number {self.cone_count}. Waiting for the third cone..."
                    )
                else:
                    # 第三次检测到锥桶时，执行任务1
                    print("Detected third cone. Executing Task 1...")
                    self.execute_task1()
                    self.cone_task_executed = True  # 标记锥桶任务已执行

    def reset_cone_detection(self):
        """重置锥桶检测状态"""
        if self.cone_detection_active:
            print("Cone detection lost. Resetting detection state...")
        self.cone_confidence_met = False  # 重置置信度状态
        self.cone_detection_active = False  # 重置活动检测状态
        self.cone_start_time = None  # 重置计时器

    def execute_task1(self):
        """执行锥桶检测任务"""
        print("Task 1: Detected a cone. Executing Task 1...")

    def execute_task2(self):
        """执行斑马线检测任务"""
        print("Task 2: Detected a zebra crossing. Executing Task 2...")
        self.executed_tasks["zebra"] = True

    def execute_task3(self):
        """执行转向标志检测任务"""
        print("Task 3: Detected a turn sign. Executing Task 3...")
        self.executed_tasks["turn_sign"] = True

    def reset_tasks(self):
        """重置任务状态"""
        for task in self.executed_tasks:
            self.executed_tasks[task] = False
        self.cone_count = 0  # 重置锥桶计数
        self.cone_task_executed = False
        self.cone_start_time = None
        self.cone_detection_active = False
        self.cone_confidence_met = False

    def get_state(self):
        """返回当前状态"""
        return self.state


def main():
    # 初始化配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
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
