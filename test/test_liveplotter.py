from odrive.utils import start_liveplotter
from src.odrive_protocol import ODriveAsciiProtocol


def get_data():
    # 假设我们从 ODrive 中获取一些数据
    odrive = ODriveAsciiProtocol(port="COM23")
    feedback = odrive.request_feedback(0)
    return feedback


start_liveplotter(get_data, legend=["Position", "Current"])


from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


