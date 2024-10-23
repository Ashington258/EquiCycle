import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import torch
import requests
import threading


class Config:
    """配置参数类"""

    MODEL_PATH = (
        "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/model/best.pt"
    )
    INPUT_SOURCE = "light.mp4"  # 可以是视频路径、摄像头ID或URL
    CONF_THRESH = 0.25
    IMG_SIZE = 1280
    ROI_TOP_LEFT_RATIO = (0, 0.35)
    ROI_BOTTOM_RIGHT_RATIO = (1, 0.95)
    MIN_BRANCH_LENGTH = 150  # 小分支过滤阈值


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


class ImageProcessor:
    """图像处理类"""

    def __init__(self, roi_top_left_ratio, roi_bottom_right_ratio, kernel_size=(5, 5)):
        self.roi_top_left_ratio = roi_top_left_ratio
        self.roi_bottom_right_ratio = roi_bottom_right_ratio
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    def define_roi(self, frame_width, frame_height):
        """定义感兴趣区域 (ROI)"""
        roi_top_left = (
            int(frame_width * self.roi_top_left_ratio[0]),
            int(frame_height * self.roi_top_left_ratio[1]),
        )
        roi_bottom_right = (
            int(frame_width * self.roi_bottom_right_ratio[0]),
            int(frame_height * self.roi_bottom_right_ratio[1]),
        )
        return roi_top_left, roi_bottom_right

    def process_mask(self, mask, frame_height, frame_width, roi, min_branch_length):
        """处理掩码，应用形态学操作和骨架提取"""
        roi_top_left, roi_bottom_right = roi

        mask_height, mask_width = mask.shape[:2]
        if (mask_height, mask_width) != (frame_height, frame_width):
            mask = cv2.resize(
                mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
            )

        mask_roi = mask[
            roi_top_left[1] : roi_bottom_right[1], roi_top_left[0] : roi_bottom_right[0]
        ]

        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, self.kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel)

        skeleton = skeletonize(mask_roi > 0).astype(np.uint8)

        # 对骨架进行标记
        labeled_skeleton = label(skeleton, connectivity=2)
        processed_skeleton = np.zeros_like(skeleton)

        for region in regionprops(labeled_skeleton):
            if region.area >= min_branch_length:
                coords = region.coords
                processed_skeleton[coords[:, 0], coords[:, 1]] = 1

                # 拟合曲线
                ys, xs = coords[:, 0], coords[:, 1]
                if len(xs) >= 2:
                    z = np.polyfit(xs, ys, 2)  # 二次多项式拟合
                    p = np.poly1d(z)
                    x_new = np.linspace(xs.min(), xs.max(), num=100)
                    y_new = p(x_new)

                    # 将拟合结果绘制在原图上
                    x_new = x_new.astype(np.int32) + roi_top_left[0]
                    y_new = y_new.astype(np.int32) + roi_top_left[1]

                    valid_indices = (
                        (x_new >= 0)
                        & (x_new < frame_width)
                        & (y_new >= 0)
                        & (y_new < frame_height)
                    )
                    x_new = x_new[valid_indices]
                    y_new = y_new[valid_indices]

                    return x_new, y_new
        return None, None


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


class VideoProcessor:
    """视频处理类，支持本地视频、摄像头或网络流"""

    def __init__(self, input_source):
        if isinstance(input_source, str):
            if input_source.startswith("http://") or input_source.startswith(
                "https://"
            ):
                self.stream = VideoStream(input_source)
                self.cap = None
            else:
                self.cap = cv2.VideoCapture(input_source)
                if not self.cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {input_source}")
                self.stream = None
        elif isinstance(input_source, int):
            self.cap = cv2.VideoCapture(input_source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开摄像头: {input_source}")
            self.stream = None
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


def main():
    # 初始化配置
    config = Config()

    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化处理器
    yolo_processor = YOLOProcessor(
        config.MODEL_PATH, config.CONF_THRESH, config.IMG_SIZE, device
    )
    image_processor = ImageProcessor(
        config.ROI_TOP_LEFT_RATIO, config.ROI_BOTTOM_RIGHT_RATIO
    )
    video_processor = VideoProcessor(config.INPUT_SOURCE)

    # 初始化计时器
    prev_time = time.time()

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        # YOLO推理
        results = yolo_processor.infer(frame)

        # 计算FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        prev_time = current_time

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

        # 定义ROI
        frame_height, frame_width = frame.shape[:2]
        roi_top_left, roi_bottom_right = image_processor.define_roi(
            frame_width, frame_height
        )

        # 绘制ROI矩形
        cv2.rectangle(
            frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2
        )  # 绿色矩形，线条粗细为2

        # 处理掩码
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255

            for mask in masks:
                x_new, y_new = image_processor.process_mask(
                    mask,
                    frame_height,
                    frame_width,
                    (roi_top_left, roi_bottom_right),
                    config.MIN_BRANCH_LENGTH,
                )
                if x_new is not None and y_new is not None:
                    # 绘制拟合曲线
                    for i in range(len(x_new) - 1):
                        cv2.line(
                            frame,
                            (x_new[i], y_new[i]),
                            (x_new[i + 1], y_new[i + 1]),
                            (0, 0, 255),
                            2,
                        )

        # 显示结果
        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    video_processor.release()


if __name__ == "__main__":
    main()
