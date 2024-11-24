import time
import torch
import cv2
from config.settings import Config
from utils.image_utils import ImageUtils
from processing.yolov8_processor import YOLOProcessor
from processing.video_stream import VideoStream
from processing.lane_fitting import LaneFitting
from visualization.display import Display


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )

    cap = cv2.VideoCapture(Config.INPUT_SOURCE)
    Display.initialize_window(
        "YOLOv8 Instance Segmentation with Centerline",
        Config.IMG_SIZE,
        int(Config.IMG_SIZE * 0.75),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = ImageUtils.resize_frame(frame, Config.IMG_SIZE)

        results = yolo_processor.infer(frame)
        # TODO: Add processing steps here

        Display.show_frame("YOLOv8 Instance Segmentation with Centerline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
