import cv2
import numpy as np


def apply_nms(results, iou_threshold=0.5):
    """应用NMS过滤边界框和分割掩膜"""
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)  # 获取类别索引
    masks = results[0].masks

    if masks is None:
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        if indices is None or len(indices) == 0:
            return [], [], None, []

        indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
        selected_indices = list(indices)
        filtered_boxes = boxes[selected_indices]
        filtered_scores = scores[selected_indices]
        filtered_classes = classes[selected_indices]  # 添加类别
        return filtered_boxes, filtered_scores, None, filtered_classes

    masks = masks.data.cpu().numpy()
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )

    if indices is None or len(indices) == 0:
        return [], [], [], []

    indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
    selected_indices = list(indices)
    filtered_boxes = boxes[selected_indices]
    filtered_scores = scores[selected_indices]
    filtered_classes = classes[selected_indices]  # 添加类别
    filtered_masks = masks[selected_indices]

    return filtered_boxes, filtered_scores, filtered_masks, filtered_classes
