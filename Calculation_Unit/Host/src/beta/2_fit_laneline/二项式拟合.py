import cv2
import time
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
import torch

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
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)

            # 骨架化处理
            skeleton = skeletonize(mask_roi > 0).astype(np.uint8)

            # 提取骨架点坐标
            points = np.column_stack(np.where(skeleton > 0))

            min_points_threshold = 20  # 设置最小点数阈值，避免拟合过短线段
            if len(points) > min_points_threshold:
                # 获取骨架点的x和y坐标
                x = points[:, 1]
                y = points[:, 0]

                # 对骨架点进行二项式拟合，degree=2表示二项式
                poly_params = np.polyfit(x, y, 2)
                poly_func = np.poly1d(poly_params)

                # 使用拟合函数计算平滑后的点
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = poly_func(x_fit)

                # 计算曲率
                first_derivative = np.polyder(poly_func, 1)
                second_derivative = np.polyder(poly_func, 2)
                curvatures = (
                    np.abs(second_derivative(x_fit))
                    / (1 + first_derivative(x_fit) ** 2) ** 1.5
                )

                # 设置最大曲率阈值，过滤掉过弯的线段
                max_curvature_threshold = 0.005  # 根据实际场景调整该值
                if np.all(curvatures < max_curvature_threshold):
                    # 曲率在合理范围内，绘制拟合曲线
                    for j in range(len(x_fit) - 1):
                        pt1 = (int(x_fit[j]), int(y_fit[j]))
                        pt2 = (int(x_fit[j + 1]), int(y_fit[j + 1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # 蓝色曲线

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
