import cv2


class ImageUtils:
    """图像处理工具类"""

    @staticmethod
    def resize_frame(frame, target_width):
        height, width = frame.shape[:2]
        scale = target_width / width
        target_height = int(height * scale)
        return cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
