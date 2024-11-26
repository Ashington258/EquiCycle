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
    INPUT_SOURCE = "dataset/video/3.mp4"  # 支持图片路径、视频路径、摄像头ID或URL
    CONF_THRESH = 0.25
    IMG_SIZE = 1280
    ROI_TOP_LEFT_RATIO = (0, 0.35)
    ROI_BOTTOM_RIGHT_RATIO = (1, 0.95)


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
        if isinstance(input_source, str):
            if input_source.startswith("http://") or input_source.startswith(
                "https://"
            ):
                self.stream = VideoStream(input_source)
                self.cap = None
            elif input_source.lower().endswith((".jpg", ".jpeg", ".png")):
                self.cap = None
                self.image = cv2.imread(input_source)
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

        if self.cap is not None:
            self.fps_original = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"原始视频帧率: {self.fps_original}")

        cv2.namedWindow(
            "YOLOv8 Instance Segmentation with Centerline", cv2.WINDOW_NORMAL
        )

    def read_frame(self):
        """读取下一帧"""
        if self.image is not None:
            return True, self.image
        if self.cap:
            ret, frame = self.cap.read()
            return ret, frame
        elif self.stream:
            frame = self.stream.get_frame()
            if frame is not None:
                return True, frame
        return False, None

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

    def is_running(self):
        return self.running


def main():
    # 初始化配置
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        config.MODEL_PATH, config.CONF_THRESH, config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(config.INPUT_SOURCE)

    prev_time = time.time()
    fps_list = []

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        # YOLO推理
        results = yolo_processor.infer(frame)

        # 创建用于绘制的副本
        annotated_frame = frame.copy()

        # 处理分割掩码并应用细化算法
        if results[0].masks is not None:
            masks = results[0].masks.data  # 获取掩码数据，形状为 (num_masks, H, W)

            for mask in masks:
                # 将掩码转换为 NumPy 数组
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # 转换为 0 和 255

                # 应用 Guo-Brown 细化算法（使用 skeletonize 近似）
                thinned_mask = skeletonize(mask_np > 0)

                # 将细化后的掩码转换为 uint8 类型
                thinned_mask_uint8 = thinned_mask.astype(np.uint8) * 255

                # 调整掩码的尺寸以匹配原始帧
                thinned_mask_resized = cv2.resize(
                    thinned_mask_uint8,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # 获取细化掩码中非零像素的索引
                indices = np.where(thinned_mask_resized > 0)

                # 在 annotated_frame 上绘制细化后的车道线（红色）
                annotated_frame[indices[0], indices[1], :] = [0, 0, 255]  # BGR

        # 计算FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        prev_time = current_time
        fps_list.append(fps)

        # 显示FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 显示结果
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
