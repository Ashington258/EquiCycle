import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import requests
import threading
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte


class Config:
    """配置参数类"""

    MODEL_PATH = "analysis/model/lane.pt"
    INPUT_SOURCE = "dataset/video/1280.mp4"  # 支持图片路径、视频路径、摄像头ID或URL
    CONF_THRESH = 0.65  # 置信度阈值
    IMG_SIZE = 640  # 输入图像宽度，保持宽高比调整

    HORIZONTAL_LINE_Y = 280  # 横线的Y坐标

    # 定义类别名称
    CLASS_NAMES = [
        "__background__",  # 替换为实际类别名
        "L 0",
        "L 1",
        "R 0",
        "R 1",
    ]


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
            "YOLOv8 Instance Segmentation with Centerline", cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(
            "YOLOv8 Instance Segmentation with Centerline",
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


def apply_nms(results, iou_threshold=0.5):
    """应用NMS过滤边界框和分割掩膜"""
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)  # 获取类别索引
    masks = results[0].masks

    if masks is None:
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        if indices is None or len(indices) == 0:
            return [], [], None, []

        indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
        selected_indices = list(indices)
        filtered_boxes = boxes[selected_indices]
        filtered_scores = scores[selected_indices]
        filtered_classes = classes[selected_indices]  # 添加类别
        return filtered_boxes, filtered_scores, None, filtered_classes

    masks = masks.data.cpu().numpy()
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )

    if indices is None or len(indices) == 0:
        return [], [], [], []

    indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
    selected_indices = list(indices)
    filtered_boxes = boxes[selected_indices]
    filtered_scores = scores[selected_indices]
    filtered_classes = classes[selected_indices]  # 添加类别
    filtered_masks = masks[selected_indices]

    return filtered_boxes, filtered_scores, filtered_masks, filtered_classes


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(Config.INPUT_SOURCE)

    prev_time = time.time()
    fps_list = []

    class_names = Config.CLASS_NAMES
    horizontal_line_y = Config.HORIZONTAL_LINE_Y  # 从配置中获取横线Y坐标

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        results = yolo_processor.infer(frame)
        filtered_boxes, filtered_scores, filtered_masks, filtered_classes = apply_nms(
            results
        )

        cv2.line(
            frame,
            (0, horizontal_line_y),
            (frame.shape[1], horizontal_line_y),
            (255, 255, 0),
            2,
        )  # 绘制横线

        intersection_points = []  # 用于存储交点

        for i, box in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = filtered_classes[i]
            score = filtered_scores[i]
            label = f"{class_names[class_id]}: {score:.2f}"

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
                            cv2.circle(
                                frame, (int(xf), int(yf)), 5, (0, 255, 0), -1
                            )  # 绘制交点
                            break

        # 计算代数平均
        if len(intersection_points) == 2:
            center_x = int((intersection_points[0][0] + intersection_points[1][0]) / 2)
            center_y = int(horizontal_line_y)

            # 绘制中心点
            cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)  # 红色中心点
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

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        fps_list.append(fps)

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

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
