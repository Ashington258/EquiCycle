class Config:
    """配置参数类"""

    # 模型路径
    LANE_MODEL = "Calculation_Unit/model/2_lane.pt"
    ELEMENTS_MODEL = "Calculation_Unit/model/elements.pt"

    # 输入源配置
    INPUT_SOURCE = "dataset/video/640.mp4"  # 支持图片路径、视频路径、摄像头ID或URL
    IMG_SIZE = 640  # 输入图像宽度，保持宽高比调整

    # 控制参数
    CONF_THRESH = 0.75  # 置信度阈值，用于设定YOLO类的基本识别阈值
    CAR_SPEED = 1  # 后轮电机速度，车辆行驶速度
    R = 250  # 调节舵机力度的参数，越大舵机力度越小
    SERVO_MIDPOINT = 960  # 舵机中值脉冲宽度

    # 目标位置参数
    HORIZONTAL_LINE_Y = 280  # 横线的Y坐标
    TARGET_X = 320  # 目标的 X 坐标（可以根据实际需求调整）
    DISTANCE_THRESHOLD = 50  # 例：50像素

    # 锥桶类参数
    CONE_TO_AVOID_INDEX = 3  # 需要避障的锥桶索引
    CONE_CONFIRMATION_DURATION = 3  # 确认锥桶所需的持续检测时间
    CONE_DET_COOLING_TIME = 13  # 锥桶检测冷却时间
    CONE_CT = 0.85  # 锥桶的置信度

    # 斑马线类
    ZEBRA_CT = 0.85  # 斑马线的置信度

    # 转向标志类
    TURN_SIGN_CT = 0.85  # 转向标志的置信度

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
