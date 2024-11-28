class Config:
    """配置参数类"""

    # MODEL_PATH = "Calculation_Unit/model/lane.pt"
    LANE_MODEL = "Calculation_Unit/model/lane.pt"
    ELEMENTS_MODEL = "Calculation_Unit/model/elements.pt"
    INPUT_SOURCE = "dataset/video/640_2.mp4"  # 支持图片路径、视频路径、摄像头ID或URL
    CONF_THRESH = 0.65  # 置信度阈值
    IMG_SIZE = 640  # 输入图像宽度，保持宽高比调整

    HORIZONTAL_LINE_Y = 280  # 横线的Y坐标
    TARGET_X = 320  # 目标的 X 坐标（可以根据实际需求调整）
    R = 250  # 调节舵机力度的参数，越大舵机力度越小
    SERVO_MIDPOINT = 960  # 舵机中值脉冲宽度
    CAR_SPEED = 1  # 后轮电机速度，车辆行驶速度
    AVOID_CONE_INDEX = 3  # 需要避障的锥桶索引
    CONE_DET_COOLING_TIME = 13  # 锥桶检测冷却时间
    # 定义类别名称
    LANE_CLASS_NAME = [
        "__background__",  # 替换为实际类别名
        "L 0",
        "L 1",
        "R 0",
        "R 1",
    ]
    ELEMENTS_CLASS_NAME = [
        "cone",
        "zebra",
        "turn_sign",
    ]
