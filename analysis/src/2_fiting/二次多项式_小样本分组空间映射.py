CLASS_NAMES = ["Lane", "Roadblock", "Zebra Crossing", "Turn Left", "Turn Right"]

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.morphology import skeletonize
from collections import defaultdict
import time


class Config:
    MODEL_PATH = "analysis/model/100_LaneSeg.pt"
    INPUT_SOURCE = "dataset/video/1280.mp4"
    CONF_THRESH = 0.45
    IMG_SIZE = 640
    ROI_TOP_LEFT_RATIO = (0, 0.5)
    ROI_BOTTOM_RIGHT_RATIO = (1, 0.95)


class Utils:
    @staticmethod
    def resize_frame(frame, target_width):
        height, width = frame.shape[:2]
        scale = target_width / width
        target_height = int(height * scale)
        return cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )

    @staticmethod
    def process_mask(mask):
        binary = (mask / 255).astype(np.uint8)
        skeleton = skeletonize(binary).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)


class YOLOProcessor:
    def __init__(self, model_path, conf_thresh, img_size, device):
        self.device = device
        self.model = YOLO(model_path).to(device)
        self.model.conf = conf_thresh
        self.model.imgsz = img_size

    def infer(self, frame):
        return self.model(frame, device=self.device, verbose=False)


class LaneDetector:
    @staticmethod
    def group_and_fit_skeleton(skeleton):
        data_points = np.column_stack(np.where(skeleton == 255))
        lines = cv2.HoughLinesP(
            skeleton, 1, np.pi / 180, threshold=50, minLineLength=87, maxLineGap=50
        )
        distance_threshold = 20
        categories = defaultdict(list)

        if lines is not None:
            for point in data_points:
                min_distance, category_label = float("inf"), -1
                for idx, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    distance = np.abs(
                        (y2 - y1) * point[1] - (x2 - x1) * point[0] + x2 * y1 - y2 * x1
                    ) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                    if distance < min_distance and distance < distance_threshold:
                        min_distance, category_label = distance, idx
                if category_label != -1:
                    categories[category_label].append(point)

        fit_results = {}
        for label, points in categories.items():
            points = np.array(points)
            x_coords, y_coords = points[:, 1], points[:, 0]
            if len(x_coords) > 2:
                fit_results[label] = np.polyfit(x_coords, y_coords, 2)

        return categories, fit_results

    @staticmethod
    def visualize_fit(frame, skeleton, categories, fit_results):
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


class VideoProcessor:
    def __init__(self, input_source):
        self.cap = cv2.VideoCapture(input_source)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开输入源: {input_source}")
        self._initialize_display_window()

    def _initialize_display_window(self):
        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Detection", Config.IMG_SIZE, int(Config.IMG_SIZE * 0.75))

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = Utils.resize_frame(frame, Config.IMG_SIZE)
        return ret, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(Config.INPUT_SOURCE)
    fps_start_time = time.time()

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        results = yolo_processor.infer(frame)
        for result in results[0].boxes:
            box, score, cls = result.xyxy, result.conf.item(), result.cls.item()
            x1, y1, x2, y2 = map(int, box[0])
            label = f"{CLASS_NAMES[int(cls)]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            skeleton_combined = np.zeros_like(masks[0], dtype=np.uint8)

            for mask in masks:
                skeleton = Utils.process_mask((mask * 255).astype(np.uint8))
                skeleton_combined = cv2.add(skeleton_combined, skeleton)

            categories, fit_results = LaneDetector.group_and_fit_skeleton(
                skeleton_combined
            )
            LaneDetector.visualize_fit(
                frame, skeleton_combined, categories, fit_results
            )

        fps = 1 / (time.time() - fps_start_time)
        fps_start_time = time.time()
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_processor.release()


if __name__ == "__main__":
    main()
