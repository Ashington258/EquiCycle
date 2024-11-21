import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 1. 读取图像
image_path = "dataset/image/Skeletonized Lane Combined.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 2. 图像二值化
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 3. 提取数据点（白色像素）
data_points = np.column_stack(np.where(binary_image == 255))

# 4. 创建标签（根据x坐标生成四个标签）
x_coords = data_points[:, 1]  # 提取x坐标
labels = (
    np.digitize(x_coords, bins=np.linspace(np.min(x_coords), np.max(x_coords), num=5))
    - 1
)  # 生成0, 1, 2, 3标签

# 5. 使用SVM进行训练，使用RBF核
clf = svm.SVC(kernel="rbf")  # 使用RBF核
clf.fit(data_points, labels)

# 6. 预测类别
predictions = clf.predict(data_points)

# 打印预测的唯一值
print("Unique predictions:", np.unique(predictions))

# 7. 可视化分类结果
visualization_image = np.ones_like(image) * 255  # 白色背景

# 将数据点绘制为不同颜色
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 四个类别的颜色
for point, prediction in zip(data_points, predictions):
    # 确保 prediction 在合法范围内
    if prediction < len(colors):
        color = colors[prediction]  # 根据预测的类别选择颜色
        cv2.circle(visualization_image, (point[1], point[0]), 1, color, -1)
    else:
        print(f"Unexpected prediction value: {prediction}")

# 8. 显示数据点的坐标
print("Data Points Coordinates (y, x):")
for point in data_points:
    print(point)

# 9. 显示数据点可视化图像，带坐标轴
plt.figure(figsize=(6, 6))
plt.title("SVM Classification of Data Points with RBF Kernel")
plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
plt.axis("on")  # 显示坐标轴
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid()  # 添加网格
plt.show()
