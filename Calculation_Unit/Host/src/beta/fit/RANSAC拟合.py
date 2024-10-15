import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
import torch
from sklearn.linear_model import RANSACRegressor

# 参数列表
params = {
    "model_path": "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/model/best.pt",  # 模型路径
    "video_path": "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/video.mp4",  # 视频文件路径
    "conf_thresh": 0.25,  # YOLOv8 模型置信度阈值
    "img_size": 1280,  # YOLOv8 模型输入图像尺寸
    "roi_top_left_ratio": (0, 0.25),  # ROI区域左上角比例
    "roi_bottom_right_ratio": (1, 0.75),  # ROI区域右下角比例
}

# 加载训练好的 YOLOv8 模型，并移动到GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(params["model_path"]).to(device)

# 设置 YOLOv8 模型的图像输入尺寸和置信度阈值
model.conf = params["conf_thresh"]
model.imgsz = params["img_size"]

# 打开视频文件或摄像头
cap = cv2.VideoCapture(params["video_path"])

# 获取视频的帧率
fps_original = cap.get(cv2.CAP_PROP_FPS)
print(f"原始视频帧率: {fps_original}")

# 视频窗口
cv2.namedWindow("YOLOv8 Instance Segmentation with Centerline", cv2.WINDOW_NORMAL)

# 初始化计时器
prev_time = time.time()

# 定义形态学核，避免在循环中重复创建
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 模型对当前帧进行推理
    results = model(frame, device=device, verbose=False)

    # 获取当前时间并计算 FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
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
        # 获取所有掩码并转换为CPU上的numpy数组
        masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255

        # 遍历每个掩码
        for mask in masks:
            # 确认掩码尺寸与帧一致
            mask_height, mask_width = mask.shape[:2]
            if (mask_height, mask_width) != (frame_height, frame_width):
                # 如果掩码尺寸不一致，按比例缩放
                scale_y = frame_height / mask_height
                scale_x = frame_width / mask_width
                mask = cv2.resize(
                    mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
                )

            # 限制处理区域到 ROI（中间区域）
            mask_roi = mask[
                roi_top_left[1] : roi_bottom_right[1],
                roi_top_left[0] : roi_bottom_right[0],
            ]

            # 形态学处理以平滑掩码
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)

            # 骨架化处理
            skeleton = skeletonize(mask_roi > 0).astype(np.uint8)

            # 提取骨架点坐标
            points = np.column_stack(np.where(skeleton > 0))

            if points.size > 0:
                # 调整点的坐标到原始图像
                points[:, 0] += roi_top_left[1]
                points[:, 1] += roi_top_left[0]

                # 使用 RANSAC 来拟合直线
                X = points[:, 1].reshape(-1, 1)  # 使用x坐标
                y = points[:, 0]  # 使用y坐标

                if len(X) > 1:  # 确保点的数量足够进行拟合
                    ransac = RANSACRegressor()
                    ransac.fit(X, y)

                    # 获取拟合直线的两个端点
                    line_start = int(X.min().item()), int(
                        ransac.predict(X.min().reshape(1, -1)).item()
                    )
                    line_end = int(X.max().item()), int(
                        ransac.predict(X.max().reshape(1, -1)).item()
                    )

                    # 在原始图像上绘制拟合的直线
                    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # 绘制分割和中心线
    annotated_frame = results[0].plot()  # 结果绘制在图像上

    # 确保annotated_frame与frame尺寸一致
    annotated_height, annotated_width = annotated_frame.shape[:2]
    if (annotated_height, annotated_width) != (frame_height, frame_width):
        annotated_frame = cv2.resize(
            annotated_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
        )

    # 使用更快的叠加方法
    combined_frame = cv2.addWeighted(frame, 0.7, annotated_frame, 0.3, 0)

    # 显示当前帧
    cv2.imshow("YOLOv8 Instance Segmentation with Centerline", combined_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放视频和窗口资源
cap.release()
cv2.destroyAllWindows()
