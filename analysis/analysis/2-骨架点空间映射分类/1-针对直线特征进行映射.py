import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image_path = "dataset/image/Skeletonized Lane Combined.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 2. 图像二值化
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 3. 提取数据点（白色像素）
data_points = np.column_stack(np.where(binary_image == 255))

# 4. 可视化数据点
visualization_image = np.ones_like(image) * 255  # 白色背景
for point in data_points:
    cv2.circle(visualization_image, (point[1], point[0]), 1, (255, 0, 0), -1)  # 蓝色点

# 5. 显示数据点的坐标
print("Data Points Coordinates (y, x):")
for point in data_points:
    print(point)

# 6. Hough变换参数空间映射
# 定义Hough参数空间的范围
theta = np.arange(-np.pi / 2, np.pi / 2, 0.01)  # 角度范围
rho_max = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
rho = np.linspace(-rho_max, rho_max, 200)  # 距离范围

# 创建Hough参数空间
H = np.zeros((len(rho), len(theta)))

# 对每个数据点进行映射
for y, x in data_points:
    for t_idx, t in enumerate(theta):
        r = x * np.cos(t) + y * np.sin(t)
        r_idx = np.argmin(np.abs(rho - r))  # 找到最近的rho索引
        H[r_idx, t_idx] += 1  # 计数

# 7. 可视化Hough参数空间
plt.figure(figsize=(12, 6))

# 显示数据点
plt.subplot(1, 2, 1)
plt.title("Data Points Visualization")
plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
plt.axis("on")  # 显示坐标轴
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 显示Hough参数空间
plt.subplot(1, 2, 2)
plt.title("Hough Parameter Space")
plt.imshow(
    H, aspect="auto", extent=[-np.pi / 2, np.pi / 2, -rho_max, rho_max], cmap="hot"
)
plt.colorbar(label="Accumulator Value")
plt.xlabel("Theta (radians)")
plt.ylabel("Rho (pixels)")
plt.axis("on")  # 显示坐标轴

plt.tight_layout()
plt.show()
