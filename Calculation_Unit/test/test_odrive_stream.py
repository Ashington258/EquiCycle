def create_control_flow(target, speed):
    # 确保速度在合理范围内
    if not isinstance(speed, int) or speed < 0 or speed > 255:
        raise ValueError("Speed must be an integer between 0 and 255.")

    # 将速度转换为 ASCII 字符
    speed_ascii = str(speed)

    # 构建控制流字符串
    control_flow = f"v {target} {speed_ascii}\n"

    return control_flow


# 示例使用
target = 1  # 控制目标
speed = 100  # 速度值
control_flow_string = create_control_flow(target, speed)
print(control_flow_string)
