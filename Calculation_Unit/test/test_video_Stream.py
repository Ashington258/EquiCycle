import cv2
import requests
import threading
import numpy as np
from time import time


class VideoStream:
    """视频流类，用于从HTTP/RTSP流中读取视频帧"""

    def __init__(self, url):
        self.url = url
        self.bytes_data = b""
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.read_stream, daemon=True)
        self.thread.start()

    def read_stream(self):
        """从流中读取视频数据"""
        response = requests.get(self.url, stream=True)
        if response.status_code != 200:
            print("无法连接到视频流")
            self.running = False
            return

        for chunk in response.iter_content(chunk_size=4096):
            self.bytes_data += chunk
            a = self.bytes_data.find(b"\xff\xd8")  # 找到JPEG图像开始标记
            b = self.bytes_data.find(b"\xff\xd9")  # 找到JPEG图像结束标记
            if a != -1 and b != -1:
                jpg = self.bytes_data[a : b + 2]
                self.bytes_data = self.bytes_data[b + 2 :]  # 继续读取下一个数据包
                self.frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )

    def get_frame(self):
        """获取当前视频帧"""
        return self.frame

    def stop(self):
        """停止视频流读取"""
        self.running = False


def test_video_stream(url):
    """测试视频流是否正常工作"""
    stream = VideoStream(url)

    # 等待视频流初始化完成
    start_time = time()
    while (
        stream.frame is None and time() - start_time < 10
    ):  # 判断 stream.frame 是否为 None
        print("正在连接视频流...")

    if stream.frame is None:  # 如果在 10 秒内未获取到视频帧
        print("未能成功连接视频流")
        return

    print(f"成功连接到视频流: {url}")

    # 显示视频流中的每一帧
    while stream.running:
        frame = stream.get_frame()
        if frame is None:
            continue

        # 显示视频帧
        cv2.imshow("Video Stream", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    stream.stop()
    cv2.destroyAllWindows()


# 测试代码入口
if __name__ == "__main__":
    video_url = "http://192.168.2.50:5000"  # 替换为你的视频流URL，支持HTTP、RTSP等流
    test_video_stream(video_url)
