import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. 读取图像
image_path = "dataset/image/Skeletonized Lane Combined.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 2. 图像二值化
# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值进行二值化
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 3. 提取数据点（白色像素）
data_points = np.column_stack(np.where(binary_image == 255))

# 4. Hough变换检测直线
lines = cv2.HoughLinesP(
    binary_image, 1, np.pi / 180, threshold=50, minLineLength=87.88, maxLineGap=50
)

# 5. 分类数据点
distance_threshold = 20  # 设定距离阈值
categories = defaultdict(list)  # 分类存储

if lines is not None:
    for point in data_points:
        min_distance = float("inf")
        category_label = -1

        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            # 计算点到直线的距离
            distance = np.abs(
                (y2 - y1) * point[1] - (x2 - x1) * point[0] + x2 * y1 - y2 * x1
            ) / (np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                category_label = idx

        # 根据分类标签存储点
        if category_label != -1:
            categories[category_label].append(point)

# 6. 可视化分类结果
colors = plt.cm.get_cmap("tab10", len(categories))  # 获取颜色映射
classified_image = np.ones_like(image) * 255  # 白色背景

for idx, (label, points) in enumerate(categories.items()):
    color = (np.array(colors(idx)[:3]) * 255).astype(int)  # 转换为BGR颜色
    for point in points:
        cv2.circle(classified_image, (point[1], point[0]), 1, tuple(color.tolist()), -1)

plt.figure(figsize=(6, 6))
plt.title("Classified Data Points")
plt.imshow(cv2.cvtColor(classified_image, cv2.COLOR_BGR2RGB))
plt.axis("on")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 7. 对每组数据点进行二次多项式拟合
fit_image = np.ones_like(image) * 255  # 白色背景
fit_results = {}  # 存储拟合结果

plt.figure(figsize=(6, 6))
plt.title("Polynomial Fit for Each Category")

for idx, (label, points) in enumerate(categories.items()):
    # 转换点坐标格式为 (x, y)
    points = np.array(points)
    x_coords = points[:, 1]  # 横坐标
    y_coords = points[:, 0]  # 纵坐标

    # 进行二次多项式拟合
    if len(x_coords) > 2:  # 至少需要3个点进行二次拟合
        poly_coeff = np.polyfit(x_coords, y_coords, 2)  # 返回拟合系数
        fit_results[label] = poly_coeff

        # 根据拟合多项式绘制曲线
        x_fit = np.linspace(x_coords.min(), x_coords.max(), 500)
        y_fit = np.polyval(poly_coeff, x_fit)  # 计算拟合曲线的y值

        # 可视化拟合曲线
        plt.plot(x_fit, y_fit, label=f"Category {label}", color=colors(idx)[:3])

        # 在背景图中绘制拟合曲线
        for i in range(len(x_fit) - 1):
            x1, y1 = int(x_fit[i]), int(y_fit[i])
            x2, y2 = int(x_fit[i + 1]), int(y_fit[i + 1])
            cv2.line(fit_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色曲线

# 显示拟合曲线图
plt.legend()
plt.axis("on")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 8. 显示拟合曲线和分类数据点的叠加结果
plt.figure(figsize=(6, 6))
plt.title("Fitted Polynomial Curves")
plt.imshow(cv2.cvtColor(fit_image, cv2.COLOR_BGR2RGB))
plt.axis("on")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 输出拟合结果
for label, coeffs in fit_results.items():
    print(f"Category {label}: Polynomial Coefficients: {coeffs}")
