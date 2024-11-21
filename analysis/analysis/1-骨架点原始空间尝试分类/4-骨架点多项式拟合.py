import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image_path = "dataset/image/Skeletonized Lane Combined.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 2. 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 二值化处理
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 4. 提取骨架点的坐标
points = np.column_stack(np.where(binary_image > 0))

# 5. 拟合多项式曲线
# 使用2次多项式拟合
if len(points) > 0:
    x = points[:, 1]
    y = points[:, 0]

    # 拟合多项式
    coefficients = np.polyfit(x, y, 3)  # 可以调整阶数
    polynomial = np.poly1d(coefficients)

    # 生成拟合曲线的x值
    x_fit = np.linspace(0, image.shape[1] - 1, 100)
    y_fit = polynomial(x_fit)

    # 6. 可视化拟合结果
    visualization_image = np.zeros_like(image)  # 创建一个黑色背景的图像
    for i in range(len(x_fit)):
        cv2.circle(
            visualization_image, (int(x_fit[i]), int(y_fit[i])), 1, (255, 0, 0), -1
        )  # 绘制拟合曲线

    # 7. 显示结果
    plt.figure(figsize=(6, 6))
    plt.title("Polynomial Curve Fitting")
    plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
