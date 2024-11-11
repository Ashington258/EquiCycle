import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize

# 参数列表
params = {
    "model_path": "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/model/best.pt",  # 模型路径
    "video_path": "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/video.mp4",  # 视频文件路径
    "conf_thresh": 0.25,  # YOLOv8 模型置信度阈值
    "img_size": 640,  # YOLOv8 模型输入图像尺寸
    "roi_top_left_ratio": (0, 0.25),  # ROI区域左上角比例
    "roi_bottom_right_ratio": (1, 0.75),  # ROI区域右下角比例
}

# 加载训练好的 YOLOv8 模型
model = YOLO(params["model_path"])

# 设置 YOLOv8 模型的图像输入尺寸和置信度阈值
model.conf = params["conf_thresh"]
model.imgsz = params["img_size"]

# 打开视频文件或摄像头
cap = cv2.VideoCapture(params["video_path"])

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"原始视频帧率: {fps}")

# 视频窗口
cv2.namedWindow("YOLOv8 Instance Segmentation with Centerline", cv2.WINDOW_NORMAL)

# 初始化计时器
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 模型对当前帧进行推理
    results = model(frame)

    # 获取当前时间
    current_time = time.time()

    # 计算 FPS
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # 将帧率写到图像上
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # 获取图像尺寸并定义 ROI 区域
    frame_height, frame_width = frame.shape[:2]
    roi_top_left = (
        int(frame_width * params["roi_top_left_ratio"][0]),
        int(frame_height * params["roi_top_left_ratio"][1]),
    )
    roi_bottom_right = (
        int(frame_width * params["roi_bottom_right_ratio"][0]),
        int(frame_height * params["roi_bottom_right_ratio"][1]),
    )

    # 在原始帧上绘制矩形框，表示感兴趣区域
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

    # 获取分割后的掩码（确保存在检测结果）
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255

        # 遍历每个掩码
        for i in range(masks.shape[0]):
            mask = masks[i]

            # 确保 mask 大小与原始帧匹配
            mask_resized = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )  # 使用最近邻插值确保掩码不失真

            # 限制处理区域到 ROI（中间区域）
            mask_roi = np.zeros_like(mask_resized)  # 创建一个全零的掩码
            mask_roi[
                roi_top_left[1] : roi_bottom_right[1],
                roi_top_left[0] : roi_bottom_right[0],
            ] = mask_resized[
                roi_top_left[1] : roi_bottom_right[1],
                roi_top_left[0] : roi_bottom_right[0],
            ]

            # 形态学处理以平滑掩码
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)

            # 骨架化处理
            skeleton = skeletonize(mask_roi > 0).astype(np.uint8)

            # 提取骨架点坐标
            points = np.column_stack(np.where(skeleton > 0))

            # 在原始帧上绘制骨架点
            for point in points:
                cv2.circle(
                    frame, (point[1], point[0]), 1, (0, 0, 255), -1
                )  # 红色表示骨架点

    # 绘制分割和中心线
    annotated_frame = results[0].plot()  # 结果绘制在图像上
    combined_frame = cv2.addWeighted(frame, 0.7, annotated_frame, 0.3, 0)

    # 显示当前帧
    cv2.imshow("YOLOv8 Instance Segmentation with Centerline", combined_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放视频和窗口资源
cap.release()
cv2.destroyAllWindows()
