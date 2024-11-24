import time
import cv2
import torch
from config.settings import Config
from models.yolo_processor import YOLOProcessor
from utils.image_processing import Utils
from utils.video_stream import VideoProcessor
from processors.nms import apply_nms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_processor = YOLOProcessor(
        Config.MODEL_PATH, Config.CONF_THRESH, Config.IMG_SIZE, device
    )
    video_processor = VideoProcessor(Config.INPUT_SOURCE)

    prev_time, fps_list = time.time(), []

    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        results = yolo_processor.infer(frame)
        filtered_boxes, filtered_scores, filtered_masks, filtered_classes = apply_nms(
            results
        )

        # 绘制推理结果（简化）
        for i, box in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = filtered_classes[i]
            color = Config.COLOR_MAP[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow("YOLOv8 Instance Segmentation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_processor.release()


if __name__ == "__main__":
    main()
