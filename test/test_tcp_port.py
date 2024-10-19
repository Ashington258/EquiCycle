import socket


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 设置超时时间
        s.settimeout(1)
        # 尝试连接本地的指定端口
        result = s.connect_ex(("127.0.0.1", port))
        # 如果返回值为0，则端口被占用
        return result == 0


if __name__ == "__main__":
    ports_to_check = [5555, 5556]
    for port in ports_to_check:
        if check_port(port):
            print(f"端口 {port} 已被占用")
        else:
            print(f"端口 {port} 可用")
