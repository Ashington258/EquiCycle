import math


class Config:
    """配置参数类"""

    # 模型路径
    LANE_MODEL = "Calculation_Unit/model/best.pt"
    ELEMENTS_MODEL = "Calculation_Unit/model/elements.pt"

    # 输入源配置
    INPUT_SOURCE = (
        "calculate_unit/Host/src/beta/640.mp4"  # 支持图片路径、视频路径、摄像头ID或URL
    )
    IMG_SIZE = 640  # 输入图像宽度，保持宽高比调整

    # 控制参数
    CONF_THRESH = 0.75  # 置信度阈值，用于设定YOLO类的基本识别阈值
    CAR_SPEED = 1  # 后轮电机速度，车辆行驶速度
    R = 250  # 调节舵机力度的参数，越大舵机力度越小
    SERVO_MIDPOINT = 960  # 舵机中值脉冲宽度
    ALPAH = 0.3  # 平滑系数，范围在 0-1，数值越小平滑程度越高

    # 目标位置参数
    HORIZONTAL_LINE_Y = 280  # 横线的Y坐标
    TARGET_X = 320  # 目标的 X 坐标（摄像头中值）
    DISTANCE_THRESHOLD = 50  # 例：50像素

    # 锥桶类参数
    CONE_TO_AVOID_INDEX = 1  # 需要避障的锥桶索引
    CONE_CONFIRMATION_DURATION = 1  # 确认锥桶所需的持续检测时间
    CONE_DET_COOLING_TIME = 13  # 锥桶检测冷却时间
    CONE_CT = 0.85  # 锥桶的置信度
    CONE_IDLE_TIME = 2  # 持续IDLE状态时间

    # 斑马线类
    ZEBRA_CT = 0.7  # 斑马线的置信度
    ZEBRA_OR_TURN_CONFIRMATION_DURATION = 0.3  # 持续监测时间
    ZEBRA_OR_TURN_IDLE_TIME = 1  # 持续IDLE状态时间
    # 转向标志类
    TURN_SIGN_CT = 0.8  # 转向标志的置信度

    # 任务类参数
    # 停车和变道类任务
    STABILIZATION_TIME = 1.5  # 停车后等待车身稳定时间
    LANE_CHANGE_ANGLE = 100  # 变道角度
    LANE_CHANGE_SPEED = -2  # 变道速度
    PARKING_TIME = 10  # 停车时间
    LANE_CHANGE_TIME = 8  # 变道时间
    # TODO 真正需要标定的恒定量,计算出来后作为恒定两
    # LANE_DISTANCE = LANE_CHANGE_SPEED * math.sin(LANE_CHANGE_ANGLE) * LANE_CHANGE_TIME

    # 避障类任务
    AVOID_SPEED = -0.5
    AVOID_ANGLE = 1

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
    # ELEMENTS_CLASS_NAME = [
    #     "zebra",
    #     "turn_sign",
    #     "cone",
    # ]


if __name__ == "__main__":
    print(Config.LANE_DISTANCE)
