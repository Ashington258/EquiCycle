import cv2
import time
from ultralytics import YOLO

# 加载训练好的 YOLOv8 模型
model = YOLO("calculate_unit/Host/src/beta/model/100_LaneSeg.pt")  # 替换为你的模型路径

# 打开视频文件或摄像头
video_path = "dataset/video/【视频】【毕导】这是科学史上最难理解的悖论.mp4"  # 替换为你的视频文件路径
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
    cv2.imshow("YOLOv8 Instance Segmentation", annotated_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
