import cv2
import time
from ultralytics import YOLO

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
cv2.namedWindow("YOLOv8 Instance Segmentation", cv2.WINDOW_NORMAL)

# 初始化计时器
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 模型对当前帧进行推理
    results = model(frame)

    # 获取当前时间并计算 FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 将帧率写到图像上
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # 遍历每个检测结果，提取 mask 并计算质心
    for result in results:
        masks = result.masks  # 获取所有对象的分割 mask
        if masks is None:
            continue

        # 遍历每个 mask，计算其质心
        for mask in masks.data:  # 访问 mask 的数据
            # 将 mask 转换为二值图像并移动到 CPU 上
            binary_mask = mask.cpu().numpy().astype("uint8") * 255

            # 计算二值图像的矩
            M = cv2.moments(binary_mask)

            # 避免除以零的情况
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # 计算质心的 X 坐标
                cy = int(M["m01"] / M["m00"])  # 计算质心的 Y 坐标

                # 在图像上绘制质心
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"({cx}, {cy})",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

    # 显示标注后的帧
    annotated_frame = results[0].plot()  # 可选：YOLO 的标注结果
    cv2.imshow("YOLOv8 Instance Segmentation", annotated_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
