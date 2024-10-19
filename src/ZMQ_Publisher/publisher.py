import zmq
import logging
import time
import json


class ZMQPublisher:
    def __init__(self, port, topic=""):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.port = port
        self.topic = topic
        self.bind()

    def bind(self):
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                logging.info(
                    f"尝试绑定到端口 tcp://*:{self.port}，尝试次数: {retry_count + 1}"
                )
                self.publisher.bind(f"tcp://*:{self.port}")
                logging.info(f"端口绑定成功: tcp://*:{self.port}")
                break
            except zmq.error.ZMQError as e:
                logging.error(f"绑定失败: {e}，重试中...")
                retry_count += 1
                time.sleep(0.5)
        if retry_count == max_retries:
            logging.error(f"无法绑定端口 tcp://*:{self.port}")
            raise RuntimeError(f"无法绑定端口 tcp://*:{self.port}")

    def send(self, message):
        if self.topic:
            self.publisher.send_multipart([self.topic.encode(), message])
        else:
            self.publisher.send(message)

    def send_json(self, data):
        if self.topic:
            self.publisher.send_multipart(
                [self.topic.encode(), json.dumps(data).encode()]
            )
        else:
            self.publisher.send_json(data)

    def send_string(self, data):
        if self.topic:
            self.publisher.send_multipart([self.topic.encode(), data.encode()])
        else:
            self.publisher.send_string(data)

    def close(self):
        self.publisher.close()
        self.context.term()
        time.sleep(1)
        logging.info(f"ZMQ上下文已终止，发布者在端口 {self.port} 已关闭")
