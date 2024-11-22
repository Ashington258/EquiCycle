import cv2
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
from skimage.morphology import skeletonize
from collections import defaultdict


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
        self.image = None
        self.cap = None
        self.stream = None

        if isinstance(input_source, str):
            if input_source.startswith(("http://", "https://")):
                self.stream = VideoStream(input_source)
            elif input_source.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image = cv2.imread(input_source)
            else:
                self.cap = cv2.VideoCapture(input_source)
                if not self.cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {input_source}")
        elif isinstance(input_source, int):
            self.cap = cv2.VideoCapture(input_source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开摄像头: {input_source}")
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
    """对二值化的车道线进行骨架化并平滑"""
    binary = (mask / 255).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)


def group_and_fit_skeleton(skeleton):
    """对骨架点进行分组并拟合"""
    data_points = np.column_stack(np.where(skeleton == 255))
    lines = cv2.HoughLinesP(
        skeleton, 1, np.pi / 180, threshold=50, minLineLength=87.88, maxLineGap=50
    )

    distance_threshold = 20
    categories = defaultdict(list)

    if lines is not None:
        for point in data_points:
            min_distance = float("inf")
            category_label = -1
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                distance = np.abs(
                    (y2 - y1) * point[1] - (x2 - x1) * point[0] + x2 * y1 - y2 * x1
                ) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if distance < min_distance and distance < distance_threshold:
                    min_distance = distance
                    category_label = idx
            if category_label != -1:
                categories[category_label].append(point)

    fit_results = {}
    for label, points in categories.items():
        points = np.array(points)
        x_coords, y_coords = points[:, 1], points[:, 0]
        if len(x_coords) > 2:
            fit_results[label] = np.polyfit(x_coords, y_coords, 2)

    return categories, fit_results


def visualize_fit(frame, skeleton, categories, fit_results):
    """可视化结果"""
    for label, points in categories.items():
        points = np.array(points)
        for point in points:
            cv2.circle(frame, (point[1], point[0]), 2, (0, 0, 255), -1)

        if label in fit_results:
            poly_coeff = fit_results[label]
            x_fit = np.linspace(points[:, 1].min(), points[:, 1].max(), 500)
            y_fit = np.polyval(poly_coeff, x_fit)
            for i in range(len(x_fit) - 1):
                x1, y1 = int(x_fit[i]), int(y_fit[i])
                x2, y2 = int(x_fit[i + 1]), int(y_fit[i + 1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(Config.INPUT_SOURCE)

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        results = yolo_processor.infer(frame)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            skeleton_combined = np.zeros_like(masks[0], dtype=np.uint8)

            for mask in masks:
                skeleton = process_skeletonization((mask * 255).astype(np.uint8))
                skeleton_combined = cv2.add(skeleton_combined, skeleton)

            categories, fit_results = group_and_fit_skeleton(skeleton_combined)
            visualize_fit(frame, skeleton_combined, categories, fit_results)

        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_processor.release()


if __name__ == "__main__":
    main()
