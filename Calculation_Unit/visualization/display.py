import cv2


class Display:
    """显示窗口相关工具"""

    @staticmethod
    def initialize_window(window_name, width, height):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    @staticmethod
    def show_frame(window_name, frame):
        cv2.imshow(window_name, frame)
