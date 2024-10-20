import zmq
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_odrive_subscriber():
    # 创建 ZeroMQ 上下文
    context = zmq.Context()

    # 创建 SUB socket
    subscriber = context.socket(zmq.SUB)

    # 连接到 ODrive 发布者（假设发布者绑定在5556端口）
    subscriber.connect("tcp://localhost:5556")

    # 订阅所有消息，使用空字符串表示订阅所有
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    logging.info("已连接到 ODrive 发布者，开始接收消息...")

    try:
        while True:
            # 接收消息
            message = subscriber.recv_string()
            logging.info(f"接收到消息: {message}")
    except KeyboardInterrupt:
        logging.info("订阅者被用户中断")
    except Exception as e:
        logging.error(f"发生错误: {e}")
    finally:
        subscriber.close()
        context.term()
        logging.info("ODrive 订阅者已关闭")


if __name__ == "__main__":
    run_odrive_subscriber()
