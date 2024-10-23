import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
import torch


class Config:
    """配置参数类"""

    MODEL_PATH = (
        "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/model/best.pt"
    )
    VIDEO_PATH = (
        "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/video.mp4"
    )
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

    def process_mask(self, mask, frame_height, frame_width, roi):
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
        points = np.column_stack(np.where(skeleton > 0))

        if points.size > 0:
            points[:, 0] += roi_top_left[1]
            points[:, 1] += roi_top_left[0]
            return points
        return np.array([])


class VideoProcessor:
    """视频处理类"""

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        self.fps_original = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"原始视频帧率: {self.fps_original}")

        cv2.namedWindow(
            "YOLOv8 Instance Segmentation with Centerline", cv2.WINDOW_NORMAL
        )

    def read_frame(self):
        """读取下一帧"""
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """释放视频资源"""
        self.cap.release()
        cv2.destroyAllWindows()


class LaneKalmanFilter:
    def __init__(self, degree=2):
        self.degree = degree
        self.A = np.eye(degree + 1)
        self.H = np.eye(degree + 1)
        self.Q = np.eye(degree + 1) * 0.01
        self.R = np.eye(degree + 1) * 0.1
        self.P = np.eye(degree + 1)
        self.x = np.zeros((degree + 1, 1))

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        y = z.reshape(-1, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(self.degree + 1) - K @ self.H) @ self.P
        return self.x.flatten()


def fit_polynomial_and_filter(points, kalman_filter, degree=2):
    if len(points) > degree:
        x = points[:, 1]
        y = points[:, 0]
        poly_coeffs = np.polyfit(x, y, degree)
        kalman_filter.predict()
        filtered_coeffs = kalman_filter.update(poly_coeffs)
        poly = np.poly1d(filtered_coeffs)
        return poly
    else:
        return None


def main():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_processor = YOLOProcessor(
        model_path=config.MODEL_PATH,
        conf_thresh=config.CONF_THRESH,
        img_size=config.IMG_SIZE,
        device=device,
    )
    image_processor = ImageProcessor(
        roi_top_left_ratio=config.ROI_TOP_LEFT_RATIO,
        roi_bottom_right_ratio=config.ROI_BOTTOM_RIGHT_RATIO,
    )
    video_processor = VideoProcessor(video_path=config.VIDEO_PATH)

    prev_time = time.time()

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        results = yolo_processor.infer(frame)
        frame_height, frame_width = frame.shape[:2]
        roi = image_processor.define_roi(frame_width, frame_height)

        lane_polynomials = []

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255

            # 创建对应数量的卡尔曼滤波器
            lane_filters = [LaneKalmanFilter(degree=2) for _ in range(len(masks))]

            for i, mask in enumerate(masks):
                points = image_processor.process_mask(
                    mask, frame_height, frame_width, roi
                )
                if points.size > 0:
                    poly = fit_polynomial_and_filter(points, lane_filters[i], degree=2)
                    if poly:
                        lane_polynomials.append(poly)
                        x_vals = np.linspace(
                            points[:, 1].min(), points[:, 1].max(), num=100
                        )
                        y_vals = poly(x_vals)
                        for j in range(1, len(x_vals)):
                            cv2.line(
                                frame,
                                (int(x_vals[j - 1]), int(y_vals[j - 1])),
                                (int(x_vals[j]), int(y_vals[j])),
                                (0, 255, 255),
                                2,
                            )

        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_processor.release()


if __name__ == "__main__":
    main()
