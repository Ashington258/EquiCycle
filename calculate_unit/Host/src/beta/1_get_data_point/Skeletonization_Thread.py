import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
import torch
import threading
import queue


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
    MAX_QUEUE_SIZE = 10  # 队列最大大小


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
            roi_top_left[1] : roi_bottom_right[1],
            roi_top_left[0] : roi_bottom_right[0],
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


def frame_reader(video_processor, frame_queue):
    """帧读取线程函数"""
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)


def frame_inference(yolo_processor, frame_queue, result_queue):
    """帧推理线程函数"""
    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)
            break
        results = yolo_processor.infer(frame)
        result_queue.put((frame, results))


def main():
    # 初始化配置
    config = Config()

    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化各个处理器
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

    # 创建队列
    frame_queue = queue.Queue(maxsize=config.MAX_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=config.MAX_QUEUE_SIZE)

    # 启动帧读取线程
    reader_thread = threading.Thread(
        target=frame_reader, args=(video_processor, frame_queue)
    )
    reader_thread.start()

    # 启动推理线程（可以根据CPU/GPU性能调整线程数量）
    infer_thread = threading.Thread(
        target=frame_inference, args=(yolo_processor, frame_queue, result_queue)
    )
    infer_thread.start()

    prev_time = time.time()

    while True:
        result = result_queue.get()
        if result is None:
            break
        frame, results = result

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
        roi_top_left, roi_bottom_right = roi

        # 绘制ROI矩形
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

        # 处理掩码
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255

            for mask in masks:
                points = image_processor.process_mask(
                    mask, frame_height, frame_width, roi
                )
                if points.size > 0:
                    # 绘制骨架点
                    frame[points[:, 0], points[:, 1]] = [0, 0, 255]  # 红色表示骨架点

        # 绘制分割结果
        annotated_frame = results[0].plot()

        # 调整annotated_frame尺寸
        annotated_height, annotated_width = annotated_frame.shape[:2]
        if (annotated_height, annotated_width) != (frame_height, frame_width):
            annotated_frame = cv2.resize(
                annotated_frame,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # 叠加分割结果
        combined_frame = cv2.addWeighted(frame, 0.7, annotated_frame, 0.3, 0)

        # 显示结果
        cv2.imshow("YOLOv8 Instance Segmentation with Centerline", combined_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 等待线程结束
    reader_thread.join()
    infer_thread.join()

    # 释放资源
    video_processor.release()


if __name__ == "__main__":
    main()
