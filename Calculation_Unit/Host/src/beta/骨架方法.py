import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize

# 加载训练好的 YOLOv8 模型
model = YOLO(
    "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/model/best.pt"
)  # 替换为你的模型路径

# 打开视频文件或摄像头
video_path = "F:/0.Temporary_Project/EquiCycle/Calculation_Unit/Host/src/beta/video.mp4"  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

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

    # 获取分割后的图像
    mask = (
        results[0].masks.data.cpu().numpy().astype(np.uint8)[0] * 255
    )  # 示例代码，请根据实际情况调整

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 骨架化
    skeleton = skeletonize(mask > 0).astype(np.uint8)

    # 提取骨架点坐标
    points = np.column_stack(np.where(skeleton > 0))

    # 绘制骨架点
    for point in points:
        cv2.circle(frame, (point[1], point[0]), 1, (0, 0, 255), -1)  # 红色表示骨架点

    # 显示分割和中心线
    annotated_frame = results[0].plot()  # 结果绘制在图像上
    combined_frame = cv2.addWeighted(frame, 0.7, annotated_frame, 0.3, 0)
    cv2.imshow("YOLOv8 Instance Segmentation with Centerline", combined_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
