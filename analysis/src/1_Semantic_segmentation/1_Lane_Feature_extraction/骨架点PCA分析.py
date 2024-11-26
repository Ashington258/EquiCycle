import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 读取图像
image_path = "dataset/image/Skeletonized Lane Combined.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 2. 二值化
# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值进行二值化，提取白色部分
_, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# 3. 读取点
# 找到白色数据点的坐标
points = np.column_stack(np.where(binary_image == 255))

# 输出数据点数量
num_points = points.shape[0]
print(f"提取的白色数据点数量: {num_points}")

# 计算数据点的空间分布特征
if num_points > 0:
    # 计算中心点
    center = np.mean(points, axis=0)
    print(f"白色数据点的中心坐标: {center}")

    # 计算点的范围
    x_range = np.max(points[:, 1]) - np.min(points[:, 1])
    y_range = np.max(points[:, 0]) - np.min(points[:, 0])
    print(f"数据点的X范围: {x_range}, Y范围: {y_range}")

# 4. PCA分析
# 执行PCA并降维至1维
pca = PCA(n_components=1)
pca_result = pca.fit_transform(points)

# 输出PCA分析结果
explained_variance = pca.explained_variance_ratio_
print(f"主成分1的方差贡献率: {explained_variance[0]:.2f}")

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(pca_result, np.zeros_like(pca_result), c="blue", marker="o", s=5)
plt.title("PCA of White Data Points (1D Projection)")
plt.xlabel("Principal Component 1")
plt.yticks([])  # 隐藏y轴刻度
plt.grid()
plt.axis("equal")
plt.show()
