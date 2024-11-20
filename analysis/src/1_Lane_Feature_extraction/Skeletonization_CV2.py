import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


class Config:
    """配置参数类"""

    MODEL_PATH = "analysis/model/100_LaneSeg.pt"
    INPUT_SOURCE = "dataset/video/1280.mp4"  # 支持图片路径、视频路径、摄像头ID或URL
    CONF_THRESH = 0.45  # 置信度阈值
    IMG_SIZE = 640  # 输入图像宽度，保持宽高比调整
    ROI_TOP_LEFT_RATIO = (0, 0.5)
    ROI_BOTTOM_RIGHT_RATIO = (1, 0.95)


class Utils:
    """通用工具类"""

    @staticmethod
    def resize_frame(frame, target_width):
        """调整帧的尺寸，保持宽高比"""
        height, width = frame.shape[:2]
        scale = target_width / width
        target_height = int(height * scale)
        return cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )


class YOLOProcessor:
    """YOLO模型处理类"""

    def __init__(self, model_path, conf_thresh, img_size, device):
        self.device = device
        self.model = YOLO(model_path).to(self.device)
        self.model.conf = conf_thresh
        self.model.imgsz = img_size

    def infer(self, frame):
        """对单帧进行推理"""
        return self.model(frame, device=self.device, verbose=False)


class VideoProcessor:
    """视频处理类，支持图片、视频、摄像头和网络流"""

    def __init__(self, input_source):
        self._initialize_input(input_source)
        if self.cap:
            self.fps_original = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"原始视频帧率: {self.fps_original}")

        # 创建可调整大小的窗口
        self._initialize_display_window()

    def _initialize_input(self, input_source):
        """初始化输入源"""
        if isinstance(input_source, str):
            if input_source.startswith(("http://", "https://")):
                self.stream = VideoStream(input_source)
                self.cap = None
                self.image = None
            elif input_source.lower().endswith((".jpg", ".jpeg", ".png")):
                self.cap = None
                self.image = cv2.imread(input_source)
                self.stream = None
            else:
                self.cap = cv2.VideoCapture(input_source)
                if not self.cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {input_source}")
                self.stream = None
                self.image = None
        elif isinstance(input_source, int):
            self.cap = cv2.VideoCapture(input_source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开摄像头: {input_source}")
            self.stream = None
            self.image = None
        else:
            raise ValueError("未知的输入源类型")

    def _initialize_display_window(self):
        """初始化显示窗口"""
        cv2.namedWindow(
            "YOLOv8 Instance Segmentation with Skeletonization", cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(
            "YOLOv8 Instance Segmentation with Skeletonization",
            Config.IMG_SIZE,
            int(Config.IMG_SIZE * 0.75),
        )

    def read_frame(self):
        """读取下一帧并调整尺寸"""
        if self.image is not None:
            frame = self.image
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return False, None
        elif self.stream:
            frame = self.stream.get_frame()
            if frame is None:
                return False, None
        else:
            return False, None

        frame = Utils.resize_frame(frame, Config.IMG_SIZE)
        return True, frame

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if self.stream:
            self.stream.stop()
        cv2.destroyAllWindows()


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


def process_skeletonization(mask):
    """对二值化的车道线进行骨架化"""
    # 归一化为[0, 1]
    binary = (mask / 255).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton


def main():
    # 初始化配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(Config.INPUT_SOURCE)

    prev_time = time.time()
    fps_list = []

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        # YOLO推理
        results = yolo_processor.infer(frame)

        # 假设结果的mask存储在results[0].masks.data
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # 假设mask格式为二值图像
            skeleton_combined = np.zeros_like(masks[0], dtype=np.uint8)
            for mask in masks:
                mask_resized = (mask * 255).astype(np.uint8)
                skeleton = process_skeletonization(mask_resized)
                skeleton_combined = cv2.add(skeleton_combined, skeleton)

            # 显示骨架化结果叠加
            cv2.imshow("Skeletonized Lane Combined", skeleton_combined)

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
        cv2.imshow("YOLOv8 Instance Segmentation with Skeletonization", annotated_frame)

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
