import time

import serial

from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


PORT = "COM27"
BAUDRATE = 460800
VELOCITY_LIMIT = 4.0
MOVE_TURNS = 10.0


def read_number(drive, property_name):
    return float(drive.read_parameter(property_name))


def check_axis_ready(drive):
    state = int(read_number(drive, "axis0.current_state"))
    axis_error = int(read_number(drive, "axis0.error"))
    motor_error = int(read_number(drive, "axis0.motor.error"))
    encoder_error = int(read_number(drive, "axis0.encoder.error"))

    if state != 8 or axis_error or motor_error or encoder_error:
        raise RuntimeError(
            "Motor 0 未就绪："
            f"state={state}, axis_error={axis_error}, "
            f"motor_error={motor_error}, encoder_error={encoder_error}"
        )


def main():
    try:
        drive = ODriveAsciiProtocol(PORT, BAUDRATE)
    except serial.SerialException as error:
        raise SystemExit(
            f"无法打开 {PORT}：{error}\n"
            "请确认端口号正确，并关闭占用该串口的程序后重试。"
        ) from error

    try:
        check_axis_ready(drive)

        position, velocity = map(
            float, drive.request_feedback(0).split()[:2]
        )
        print(f"当前 Motor 0 位置：{position:.3f} 圈")
        print(f"当前 Motor 0 速度：{velocity:.3f} 圈/秒")

        old_input_mode = int(
            read_number(drive, "axis0.controller.config.input_mode")
        )
        print(f"当前 input_mode：{old_input_mode}")

        print("先把位置目标设为当前位置，避免遗留目标在切换模式后生效。")
        drive.motor_position(0, position, velocity_lim=VELOCITY_LIMIT)
        time.sleep(0.2)

        safe_target = read_number(drive, "axis0.controller.input_pos")
        if abs(safe_target - position) > 0.05:
            raise RuntimeError(
                f"安全目标校验失败：encoder={position}, input_pos={safe_target}；"
                "未切换输入模式或发送 10 圈目标。"
            )

        if old_input_mode != 3:
            drive.write_parameter(
                "axis0.controller.config.input_mode", 3
            )
            time.sleep(0.2)

        check_axis_ready(drive)
        control_mode = int(
            read_number(drive, "axis0.controller.config.control_mode")
        )
        input_mode = int(
            read_number(drive, "axis0.controller.config.input_mode")
        )
        input_position = read_number(drive, "axis0.controller.input_pos")
        position, velocity = map(
            float, drive.request_feedback(0).split()[:2]
        )

        print(
            f"配置检查：control_mode={control_mode}, "
            f"input_mode={input_mode}, input_pos={input_position:.3f}, "
            f"encoder_pos={position:.3f}"
        )
        if control_mode != 3 or input_mode != 3:
            raise RuntimeError(
                "驱动器没有进入位置控制/位置滤波模式；未发送运动目标。"
            )
        if abs(input_position - position) > 0.1:
            raise RuntimeError(
                "位置目标与编码器位置差异过大；未发送运动目标。"
            )

        confirmation = input(
            "确认电机及机构已固定、运动范围安全后，输入 MOVE 开始转动 10 圈："
        )
        if confirmation != "MOVE":
            print("未输入 MOVE，未发送 10 圈运动命令。")
            return

        check_axis_ready(drive)
        input_mode = int(
            read_number(drive, "axis0.controller.config.input_mode")
        )
        if input_mode != 3:
            raise RuntimeError(
                f"input_mode 已变化为 {input_mode}；未发送运动目标。"
            )

        position, velocity = map(
            float, drive.request_feedback(0).split()[:2]
        )
        target = position + MOVE_TURNS
        print(
            f"发送目标：当前位置 {position:.3f} 圈 -> "
            f"{target:.3f} 圈；速度上限 {VELOCITY_LIMIT:.1f} 圈/秒"
        )
        drive.motor_position(
            0, target, velocity_lim=VELOCITY_LIMIT
        )

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            time.sleep(0.2)
            position, velocity = map(
                float, drive.request_feedback(0).split()[:2]
            )
            print(
                f"位置：{position:.3f} 圈；速度：{velocity:.3f} 圈/秒"
            )
            if abs(position - target) < 0.05 and abs(velocity) < 0.1:
                print("已到达目标位置。")
                return

        print(
            "30 秒内未检测到到位。目标仍可能保留在驱动器中；"
            "不要切换控制模式或重复运行，请先检查驱动器状态。"
        )
    finally:
        drive.close()


if __name__ == "__main__":
    main()
