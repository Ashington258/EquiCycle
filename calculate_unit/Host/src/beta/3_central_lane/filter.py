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
    MODEL_PATH = "calculate_unit/Host/src/beta/model/100_LaneSeg.pt"
    INPUT_SOURCE = "dataset/video/完整测试视频.mp4"  # 可以是视频路径、摄像头ID或URL
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

        lane_curves = []  # 用于存储每条车道线的拟合曲线参数

        for region in regionprops(labeled_skeleton):
            if region.area >= min_branch_length:
                coords = region.coords
                processed_skeleton[coords[:, 0], coords[:, 1]] = 1

                # 拟合曲线
                ys, xs = coords[:, 0], coords[:, 1]
                if len(xs) >= 2:
                    z = np.polyfit(ys, xs, 2)  # 二次多项式拟合
                    lane_curves.append(z)  # 存储多项式系数

        return lane_curves


class VideoProcessor:
    """视频处理类，支持本地视频、摄像头或网络流"""

    def __init__(self, input_source):
        if isinstance(input_source, str):
            if input_source.startswith("http://") or input_source.startswith("https://"):
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

    # 初始化历史中心线列表
    centerline_history = []

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
        )  # 保持绿色矩形，线条粗细为2

        # 绘制检测窗口内的中线
        roi_mid_x = (roi_top_left[0] + roi_bottom_right[0]) // 2  # ROI中线的X坐标
        cv2.line(frame, (roi_mid_x, roi_top_left[1]), (roi_mid_x, roi_bottom_right[1]), (0, 255, 0), 2)

        # 存储所有拟合的车道线曲线参数
        lane_curves = []

        # 处理掩码
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255

            for mask in masks:
                curves = image_processor.process_mask(
                    mask,
                    frame_height,
                    frame_width,
                    (roi_top_left, roi_bottom_right),
                    config.MIN_BRANCH_LENGTH,
                )
                if curves:
                    lane_curves.extend(curves)

        if lane_curves:
            # 对车道线进行排序
            lane_curves_sorted = []
            for curve in lane_curves:
                # 获取曲线在ROI底部的x坐标
                y_bottom = roi_bottom_right[1] - roi_top_left[1]
                x_bottom = np.polyval(curve, y_bottom)
                lane_curves_sorted.append((curve, x_bottom))

            # 按x坐标排序
            lane_curves_sorted.sort(key=lambda item: item[1])
            sorted_curves = [item[0] for item in lane_curves_sorted]

            # 计算水平距离和中心线
            distances = []
            for i in range(len(sorted_curves) - 1):
                curve_left = sorted_curves[i]
                curve_right = sorted_curves[i + 1]

                y_vals = np.linspace(0, roi_bottom_right[1] - roi_top_left[1], num=100)
                x_vals_left = np.polyval(curve_left, y_vals)
                x_vals_right = np.polyval(curve_right, y_vals)

                # 计算中心线
                x_vals_center = (x_vals_left + x_vals_right) / 2

                # 平滑处理
                if centerline_history:
                    previous_centerline = centerline_history[-1]
                    deviation = np.abs(x_vals_center - previous_centerline)
                    threshold = 10  # 设定跳变阈值
                    smoothing_factor = np.where(deviation > threshold, 0.7, 0.3)
                    x_vals_center = (1 - smoothing_factor) * previous_centerline + smoothing_factor * x_vals_center

                # 更新中心线历史
                centerline_history.append(x_vals_center)
                if len(centerline_history) > 5:  # 只保留最近5帧的历史记录
                    centerline_history.pop(0)

                # 计算水平距离并更新列表
                distances.append(x_vals_center - roi_mid_x)

                x_vals_center = x_vals_center.astype(np.int32) + roi_top_left[0]
                y_vals_center = y_vals.astype(np.int32) + roi_top_left[1]

                valid_indices = (
                    (x_vals_center >= 0)
                    & (x_vals_center < frame_width)
                    & (y_vals_center >= 0)
                    & (y_vals_center < frame_height)
                )
                x_vals_center = x_vals_center[valid_indices]
                y_vals_center = y_vals_center[valid_indices]

                # 绘制平滑后的中心线
                for j in range(len(x_vals_center) - 1):
                    cv2.line(
                        frame,
                        (x_vals_center[j], y_vals_center[j]),
                        (x_vals_center[j + 1], y_vals_center[j + 1]),
                        (255, 0, 0),
                        2,
                    )

            # 绘制车道线
            for curve in sorted_curves:
                y_vals = np.linspace(0, roi_bottom_right[1] - roi_top_left[1], num=100)
                x_vals = np.polyval(curve, y_vals)

                x_vals = x_vals.astype(np.int32) + roi_top_left[0]
                y_vals = y_vals.astype(np.int32) + roi_top_left[1]

                valid_indices = (
                    (x_vals >= 0)
                    & (x_vals < frame_width)
                    & (y_vals >= 0)
                    & (y_vals < frame_height)
                )
                x_vals = x_vals[valid_indices]
                y_vals = y_vals[valid_indices]

                # 绘制拟合曲线（红色车道线）
                for i in range(len(x_vals) - 1):
                    cv2.line(
                        frame,
                        (x_vals[i], y_vals[i]),
                        (x_vals[i + 1], y_vals[i + 1]),
                        (0, 0, 255),
                        2,
                    )

            # 计算并显示平均水平距离
            if distances:
                avg_distance = np.mean([np.mean(dist) for dist in distances])
                cv2.putText(
                    frame,
                    f"Avg Distance: {avg_distance:.2f}px",
                    (frame_width - 400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
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
