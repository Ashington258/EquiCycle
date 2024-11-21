import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# 4. 使用K-Means进行聚类
n_clusters = 3  # 设定聚类数量，可以根据需要进行调整
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data_points)

# 5. 获取聚类标签
labels = kmeans.labels_

# 6. 可视化聚类结果
# 创建一个白色背景的图像用于可视化
visualization_image = np.ones_like(image) * 255  # 白色背景

# 将数据点绘制为不同颜色
colors = plt.cm.get_cmap("hsv", n_clusters)  # 使用HSV颜色映射
for point, label in zip(data_points, labels):
    color = colors(label)[:3]  # 获取颜色，忽略alpha通道
    cv2.circle(
        visualization_image,
        (point[1], point[0]),
        1,
        (color[2] * 255, color[1] * 255, color[0] * 255),
        -1,
    )  # 颜色转换为BGR格式

# 7. 显示数据点的坐标
print("Data Points Coordinates (y, x):")
for point in data_points:
    print(point)

# 8. 显示数据点可视化图像，带坐标轴
plt.figure(figsize=(6, 6))
plt.title("K-Means Clustering of Data Points")
plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
plt.axis("on")  # 显示坐标轴
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
