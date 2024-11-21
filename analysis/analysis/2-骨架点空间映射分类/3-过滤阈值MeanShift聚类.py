import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image_path = "dataset/image/three_line.png"  # 使用绝对路径
print(f"Trying to load image from: {image_path}")
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# 2. 图像二值化
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 3. 提取数据点（白色像素）
data_points = cv2.findNonZero(binary_image)
if data_points is not None:
    data_points = data_points.reshape(-1, 2)  # (x, y) 格式
else:
    raise ValueError("No data points found in the binary image.")

# 4. 可视化原始数据点
visualization_image = np.ones_like(image) * 255  # 白色背景
for point in data_points:
    cv2.circle(visualization_image, tuple(point), 1, (255, 0, 0), -1)  # 蓝色点

# 5. Hough变换参数空间映射
theta = np.arange(-np.pi / 2, np.pi / 2, 0.01)  # 角度范围
rho_max = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
rho = np.linspace(-rho_max, rho_max, 200)  # 距离范围

# 向量化计算 rho 值
x_coords, y_coords = data_points[:, 0], data_points[:, 1]
rho_vals = np.outer(x_coords, np.cos(theta)) + np.outer(y_coords, np.sin(theta))

# 将 rho 值映射到索引
rho_indices = (
    (rho_vals - rho.min()) / (rho.max() - rho.min()) * (len(rho) - 1)
).astype(int)

# 创建 Hough 参数空间
H = np.zeros((len(rho), len(theta)), dtype=np.int32)
for col in range(rho_vals.shape[1]):
    np.add.at(H[:, col], rho_indices[:, col], 1)

# 6. 密度过滤
density_threshold = 150  # 设置密度阈值
filtered_indices = np.argwhere(H > density_threshold)

# 7. 重新绘制筛选后的数据点
filtered_image = np.ones_like(image) * 255  # 白色背景
filtered_points = []

for r_idx, t_idx in filtered_indices:
    r = rho[r_idx]
    t = theta[t_idx]
    for y, x in data_points:
        if np.isclose(x * np.cos(t) + y * np.sin(t), r, atol=1.0):  # 容差为1.0
            filtered_points.append((x, y))

for point in filtered_points:
    cv2.circle(filtered_image, point[::-1], 1, (0, 255, 0), -1)  # 绿色点

# 8. 可视化
plt.figure(figsize=(12, 6))

# 原始数据点可视化
plt.subplot(1, 3, 1)
plt.title("Original Data Points")
plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
plt.axis("on")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Hough 参数空间可视化
plt.subplot(1, 3, 2)
plt.title("Hough Parameter Space")
plt.imshow(
    H,
    aspect="auto",
    extent=[-np.pi / 2, np.pi / 2, -rho_max, rho_max],
    cmap="hot",
)
plt.colorbar(label="Accumulator Value")
plt.xlabel("Theta (radians)")
plt.ylabel("Rho (pixels)")
plt.axis("on")

# 筛选后的点云可视化
plt.subplot(1, 3, 3)
plt.title("Filtered Points Visualization")
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.axis("on")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.tight_layout()
plt.show()
