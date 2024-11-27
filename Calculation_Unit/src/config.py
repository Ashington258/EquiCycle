class Config:
    """配置参数类"""

    MODEL_PATH = "Calculation_Unit/model/lane.pt"
    INPUT_SOURCE = "http://192.168.2.50:5000/"  # 支持图片路径、视频路径、摄像头ID或URL
    CONF_THRESH = 0.65  # 置信度阈值
    IMG_SIZE = 640  # 输入图像宽度，保持宽高比调整

    HORIZONTAL_LINE_Y = 280  # 横线的Y坐标
    TARGET_X = 320  # 目标的 X 坐标（可以根据实际需求调整）
    R = 250  # 调节舵机力度的参数，越大舵机力度越小
    SERVO_MIDPOINT = 960  # 舵机中值脉冲宽度

    # 定义类别名称
    CLASS_NAMES = [
        "__background__",  # 替换为实际类别名
        "L 0",
        "L 1",
        "R 0",
        "R 1",
    ]
