import psutil
import time


def check_port_usage(port):
    """检查指定端口的使用情况"""
    connections = psutil.net_connections(kind="inet")
    for conn in connections:
        if conn.laddr.port == port:
            return True, conn.raddr  # 返回是否占用以及远程地址
    return False, None


def monitor_ports(ports, interval=5):
    """监测指定端口的使用情况"""
    while True:
        for port in ports:
            is_used, remote_address = check_port_usage(port)
            if is_used:
                print(f"端口 {port} 被占用，远程地址: {remote_address}")
            else:
                print(f"端口 {port} 可用")

        # 这里可以添加数据吞吐量的监测逻辑
        # 假设我们使用一个简单的计数器来模拟数据吞吐量
        # 实际应用中可能需要结合您的数据发送/接收逻辑
        print("数据吞吐量监测: ...")  # 这里可以替换为实际的吞吐量计算

        time.sleep(interval)


if __name__ == "__main__":
    ports_to_monitor = [5555, 5556]
    print("开始监测端口使用情况...")
    monitor_ports(ports_to_monitor)
