import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
from skimage.morphology import skeletonize
from sklearn.neighbors import LocalOutlierFactor


class Config:
    """配置参数类"""

    MODEL_PATH = "analysis/model/equicycle.pt"
    INPUT_SOURCE = "dataset/video/1280.mp4"  # 支持图片路径、视频路径、摄像头ID或URL
    CONF_THRESH = 0.65  # 置信度阈值
    IMG_SIZE = 640  # 输入图像宽度，保持宽高比调整
    CLASS_NAMES = ["Lane", "Roadblock", "Zebra Crossing", "Turn Left", "Turn Right"]
    COLOR_MAP = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


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
    """视频处理类"""

    def __init__(self, input_source):
        self._initialize_input(input_source)
        if self.cap:
            print(f"原始视频帧率: {self.cap.get(cv2.CAP_PROP_FPS):.2f}")
        self._initialize_display_window()

    def _initialize_input(self, input_source):
        """初始化输入源"""
        if isinstance(input_source, str):
            if input_source.startswith(("http://", "https://")):
                self.stream = VideoStream(input_source)
                self.cap, self.image = None, None
            elif input_source.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image = cv2.imread(input_source)
                self.cap, self.stream = None, None
            else:
                self.cap = cv2.VideoCapture(input_source)
                if not self.cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {input_source}")
                self.stream, self.image = None, None
        elif isinstance(input_source, int):
            self.cap = cv2.VideoCapture(input_source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开摄像头: {input_source}")
            self.stream, self.image = None, None
        else:
            raise ValueError("未知的输入源类型")

    def _initialize_display_window(self):
        """初始化显示窗口"""
        cv2.namedWindow(
            "YOLOv8 Instance Segmentation with Centerline", cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(
            "YOLOv8 Instance Segmentation with Centerline",
            Config.IMG_SIZE,
            int(Config.IMG_SIZE * 0.75),
        )

    def read_frame(self):
        """读取下一帧并调整尺寸"""
        frame = (
            self.image
            if self.image is not None
            else (
                self.stream.get_frame()
                if self.stream
                else (self.cap.read()[1] if self.cap else None)
            )
        )
        return (frame is not None), (
            Utils.resize_frame(frame, Config.IMG_SIZE)
            if frame is not None
            else (False, None)
        )

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
        threading.Thread(target=self.read_stream, daemon=True).start()

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


def apply_nms(results, iou_threshold=0.4):
    """应用NMS过滤边界框和分割掩膜"""
    boxes, scores, classes = (
        results[0].boxes.xyxy.cpu().numpy(),
        results[0].boxes.conf.cpu().numpy(),
        results[0].boxes.cls.cpu().numpy().astype(int),
    )
    masks = (
        results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
    )
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if indices is None or len(indices) == 0:
        return [], [], masks, []
    selected_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
    return (
        boxes[selected_indices],
        scores[selected_indices],
        (masks[selected_indices] if masks is not None else None),
        classes[selected_indices],
    )


def fit_lane_points_and_draw(frame, skeleton, color=(0, 0, 255)):
    """对骨架点拟合三次多项式并绘制拟合曲线，去除离群点"""
    y_coords, x_coords = np.where(skeleton > 0)
    if len(x_coords) < 4:  # 如果点太少，跳过处理
        return frame

    points = np.column_stack((y_coords, x_coords))
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
    is_inlier = lof.fit_predict(points) == 1

    filtered_points = points[is_inlier]
    if len(filtered_points) < 4:
        return frame

    y_coords, x_coords = filtered_points[:, 0], filtered_points[:, 1]
    poly_func = np.poly1d(np.polyfit(y_coords, x_coords, 3))

    for y, x in zip(
        np.linspace(min(y_coords), max(y_coords), num=500).astype(int),
        poly_func(np.linspace(min(y_coords), max(y_coords), num=500)).astype(int),
    ):
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            frame[y, x] = color

    return frame


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(Config.INPUT_SOURCE)

    prev_time, fps_list = time.time(), []

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        results = yolo_processor.infer(frame)
        filtered_boxes, filtered_scores, filtered_masks, filtered_classes = apply_nms(
            results
        )

        for i, box in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id, score = filtered_classes[i], filtered_scores[i]
            label = f"{Config.CLASS_NAMES[class_id]}: {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.COLOR_MAP[class_id], 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            if class_id == 0 and filtered_masks is not None:
                lane_mask = cv2.resize(
                    filtered_masks[i],
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                lane_binary_mask = lane_mask > 0
                skeleton = skeletonize(lane_binary_mask)
                frame[skeleton] = (0, 255, 255)
                frame = fit_lane_points_and_draw(frame, skeleton)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        fps_list.append(fps)

        cv2.putText(
            frame,
            f"FPS: {np.mean(fps_list[-20:]):.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_processor.release()


if __name__ == "__main__":
    main()
