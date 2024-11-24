class Config:
    MODEL_PATH = "analysis/model/equicycle.pt"
    INPUT_SOURCE = "dataset/video/1280.mp4"
    CONF_THRESH = 0.65
    IMG_SIZE = 640
    ROI_TOP_LEFT_RATIO = (0, 0.35)
    ROI_BOTTOM_RIGHT_RATIO = (1, 0.95)
    CLASS_NAMES = ["Lane", "Roadblock", "Zebra Crossing", "Turn Left", "Turn Right"]
    COLOR_MAP = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
