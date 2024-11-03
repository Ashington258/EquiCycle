import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from sklearn.linear_model import RANSACRegressor
import torch


class Config:
    """配置参数类"""

    MODEL_PATH = (
        "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/model/best.pt"
    )
    VIDEO_PATH = "1.mp4"
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

        # 调整掩码尺寸
        mask_height, mask_width = mask.shape[:2]
        if (mask_height, mask_width) != (frame_height, frame_width):
            mask = cv2.resize(
                mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
            )

        # 限制处理区域到 ROI
        mask_roi = mask[
            roi_top_left[1] : roi_bottom_right[1], roi_top_left[0] : roi_bottom_right[0]
        ]

        # 形态学处理以平滑掩码
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, self.kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel)

        # 骨架化处理
        skeleton = skeletonize(mask_roi > 0).astype(np.uint8)

        # 提取骨架点坐标
        points = np.column_stack(np.where(skeleton > 0))

        if points.size > 0:
            # 调整点的坐标到原始图像
            points[:, 0] += roi_top_left[1]
            points[:, 1] += roi_top_left[0]
            return points
        return np.array([])

    def fit_lane_with_ransac(self, points):
        """使用RANSAC拟合平滑车道线"""
        if len(points) > 0:
            # 拆分为 x 和 y 坐标
            X = points[:, 1].reshape(-1, 1)  # x 坐标
            y = points[:, 0]  # y 坐标

            # 使用 RANSAC 进行拟合
            ransac = RANSACRegressor()
            ransac.fit(X, y)
            line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            line_y = ransac.predict(line_x).astype(int)

            return np.hstack((line_y.reshape(-1, 1), line_x))
        else:
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


def main():
    # 初始化配置
    config = Config()

    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化各个处理器
    yolo_processor = YOLOProcessor(
        config.MODEL_PATH, config.CONF_THRESH, config.IMG_SIZE, device
    )
    image_processor = ImageProcessor(
        config.ROI_TOP_LEFT_RATIO, config.ROI_BOTTOM_RIGHT_RATIO
    )
    video_processor = VideoProcessor(config.VIDEO_PATH)

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

        # 在图像上显示FPS
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
        roi = image_processor.define_roi(frame_width, frame_height)

        # 绘制ROI矩形
        cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)

        # 处理掩码
        lane_lines = []  # 存储每条车道线的点
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255

            for mask in masks:
                points = image_processor.process_mask(
                    mask, frame_height, frame_width, roi
                )
                if points.size > 0:
                    # 使用RANSAC拟合并获取平滑车道线点
                    lane_line = image_processor.fit_lane_with_ransac(points)
                    if lane_line.size > 0:
                        lane_lines.append(lane_line)

        if len(lane_lines) >= 2:
            # 根据车道线的平均 x 坐标对其进行排序
            lane_lines.sort(key=lambda line: np.mean(line[:, 1]))

            centerlines = []

            # 计算相邻车道线的中心线
            for i in range(len(lane_lines) - 1):
                line1 = lane_lines[i]
                line2 = lane_lines[i + 1]

                # 找到两个车道线 y 范围的交集
                y_min = max(line1[:, 0].min(), line2[:, 0].min())
                y_max = min(line1[:, 0].max(), line2[:, 0].max())

                if y_max <= y_min:
                    continue  # 如果没有重叠区域，跳过

                # 在重叠的 y 范围内生成 y 值
                y_values = np.linspace(y_min, y_max, num=100)

                # 使用线性插值计算对应的 x 值
                x1 = np.interp(y_values, line1[:, 0], line1[:, 1])
                x2 = np.interp(y_values, line2[:, 0], line2[:, 1])

                # 计算中点
                x_mid = (x1 + x2) / 2
                y_mid = y_values

                centerline_pts = np.vstack((y_mid, x_mid)).T.astype(int)
                centerlines.append(centerline_pts)

                # 绘制中心线
                for j in range(1, len(centerline_pts)):
                    cv2.line(
                        frame,
                        (centerline_pts[j - 1][1], centerline_pts[j - 1][0]),
                        (centerline_pts[j][1], centerline_pts[j][0]),
                        (255, 0, 0),
                        2,
                    )

        # 绘制车道线
        for lane_line in lane_lines:
            for k in range(1, len(lane_line)):
                cv2.line(
                    frame,
                    (int(lane_line[k - 1][1]), int(lane_line[k - 1][0])),
                    (int(lane_line[k][1]), int(lane_line[k][0])),
                    (0, 255, 255),
                    2,
                )

        # 显示结果
        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    video_processor.release()


if __name__ == "__main__":
    main()
