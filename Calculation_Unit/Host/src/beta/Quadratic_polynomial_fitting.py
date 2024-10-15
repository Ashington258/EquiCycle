import cv2
import time
import numpy as np
from ultralytics import YOLO

# 加载训练好的 YOLOv8 模型

model = YOLO(
    "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/model/best.pt"
)  # 替换为你的模型路径

# 打开视频文件或摄像头
video_path = "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/video.mp4"  # 替换为你的视频文件路
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"原始视频帧率: {fps}")

# 视频窗口
cv2.namedWindow("YOLOv8 Instance Segmentation", cv2.WINDOW_NORMAL)

# 初始化计时器
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 获取图像尺寸
    height, width = frame.shape[:2]

    # YOLOv8 模型对当前帧进行推理
    results = model(frame)

    # 获取当前时间
    current_time = time.time()

    # 计算 FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 将帧率写到图像上
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # 获取分割后的图像并显示
    annotated_frame = results[0].plot()  # 结果绘制在图像上

    # 提取所有分割的 masks
    masks = results[0].masks.data.cpu().numpy()

    for i, mask in enumerate(masks):
        # 确保 mask 的尺寸与原始图像一致
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height))

        # 将 mask 转换为二值图像
        binary_mask = (mask > 0.5).astype(np.uint8)  # 阈值化，生成二值 mask

        # 计算当前 mask 的矩
        M = cv2.moments(binary_mask)

        if M["m00"] != 0:  # 防止除以0的情况
            # 计算质心坐标
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 在图像上绘制质心
            cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)

            # 获取 mask 中所有非零的像素点坐标，用于拟合
            points = np.column_stack(np.where(binary_mask > 0))

            if points.shape[0] > 0:
                # 提取 x, y 坐标
                x = points[:, 1]  # 列坐标
                y = points[:, 0]  # 行坐标

                # 使用 NumPy 拟合二次曲线 y = ax^2 + bx + c
                poly_params = np.polyfit(x, y, 2)  # 拟合二次多项式

                # 生成拟合曲线的 x 点
                x_fit = np.linspace(np.min(x), np.max(x), num=100)

                # 根据多项式参数生成 y_fit
                y_fit = (
                    poly_params[0] * x_fit**2 + poly_params[1] * x_fit + poly_params[2]
                )

                # 限制拟合曲线在图像范围内
                y_fit = np.clip(y_fit, 0, height)
                x_fit = np.clip(x_fit, 0, width)

                # 在图像上绘制拟合的车道线
                for j in range(len(x_fit) - 1):
                    pt1 = (int(x_fit[j]), int(y_fit[j]))
                    pt2 = (int(x_fit[j + 1]), int(y_fit[j + 1]))
                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)

        else:
            cX, cY = 0, 0

    # 显示处理后的图像
    cv2.imshow("YOLOv8 Instance Segmentation", annotated_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
