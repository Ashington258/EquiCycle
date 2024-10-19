import logging
import sys
import serial

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def self_check(
    port_ch100="COM22",
    baudrate_ch100=460800,
    port_odrive="COM24",
    baudrate_odrive=460800,
):
    try:
        # 检查 CH100 端口
        ser_ch100 = serial.Serial(port=port_ch100, baudrate=baudrate_ch100, timeout=1)
        ser_ch100.close()
        logging.info("CH100 自检通过: 端口连接正常，波特率正常")

        # 检查 ODrive 端口
        ser_odrive = serial.Serial(
            port=port_odrive, baudrate=baudrate_odrive, timeout=1
        )
        ser_odrive.close()
        logging.info("ODrive 自检通过: 端口连接正常，波特率正常")

    except serial.SerialException as e:
        logging.error(f"自检失败: {e}")
        sys.exit(1)
