import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image_path = "dataset/image/Skeletonized Lane Combined.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 2. 图像二值化
# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值进行二值化
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 3. 提取数据点（白色像素）
# 找到白色像素的坐标
data_points = np.column_stack(np.where(binary_image == 255))

# 4. 可视化数据点
# 创建一个白色背景的图像用于可视化
visualization_image = np.ones_like(image) * 255  # 白色背景

# 将白色数据点绘制为蓝色
for point in data_points:
    cv2.circle(visualization_image, (point[1], point[0]), 1, (255, 0, 0), -1)  # 蓝色点

# 5. 显示数据点的坐标
print("Data Points Coordinates (y, x):")
for point in data_points:
    print(point)

# 6. 显示数据点可视化图像，带坐标轴
plt.figure(figsize=(6, 6))
plt.title("Data Points Visualization")
plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
plt.axis("on")  # 显示坐标轴
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
