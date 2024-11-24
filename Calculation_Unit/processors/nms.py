import cv2
import numpy as np


def apply_nms(results, iou_threshold=0.4):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    masks = (
        results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
    )

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if indices is None or len(indices) == 0:
        return [], [], masks, []
    selected_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
    return (
        boxes[selected_indices],
        scores[selected_indices],
        masks[selected_indices],
        classes[selected_indices],
    )
