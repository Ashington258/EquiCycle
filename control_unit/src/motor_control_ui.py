import sys
import time

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


AXIS = 0
BAUDRATE = 460800
MAX_SPEED_REV_S = 5.0


class MotorControlUI(QWidget):
    def __init__(self):
        super().__init__()
        self.odrive = None
        self.armed = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("ODrive 电机手动控制 · AXIS0")
        self.setMinimumSize(680, 560)
        self.resize(720, 600)

        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(15, 23, 42))
        dark_palette.setColor(QPalette.WindowText, QColor(226, 232, 240))
        dark_palette.setColor(QPalette.Base, QColor(30, 41, 59))
        dark_palette.setColor(QPalette.Text, QColor(241, 245, 249))
        dark_palette.setColor(QPalette.Button, QColor(51, 65, 85))
        dark_palette.setColor(QPalette.ButtonText, QColor(248, 250, 252))
        dark_palette.setColor(QPalette.Highlight, QColor(14, 165, 233))
        dark_palette.setColor(QPalette.HighlightedText, QColor(248, 250, 252))
        QApplication.setPalette(dark_palette)
        QApplication.setFont(QFont("Microsoft YaHei UI", 11))

        self.setStyleSheet(
            """
            QWidget { color: #e2e8f0; }
            QGroupBox {
                background: #1e293b;
                border: 1px solid #475569;
                border-radius: 10px;
                font-weight: 600;
                margin-top: 14px;
                padding: 18px 14px 14px 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #f8fafc;
            }
            QLineEdit {
                background: #0f172a;
                border: 1px solid #64748b;
                border-radius: 6px;
                color: #f8fafc;
                min-height: 42px;
                padding: 0 12px;
                selection-background-color: #0284c7;
            }
            QLineEdit:focus { border: 2px solid #38bdf8; }
            QPushButton {
                background: #334155;
                border: 1px solid #64748b;
                border-radius: 7px;
                min-height: 44px;
                padding: 0 18px;
                font-weight: 600;
            }
            QPushButton:hover { background: #475569; }
            QPushButton:pressed { background: #1e293b; }
            QPushButton:focus { border: 2px solid #7dd3fc; }
            QPushButton:disabled { color: #64748b; background: #1e293b; }
            QPushButton#connectButton { background: #0369a1; border-color: #38bdf8; }
            QPushButton#connectButton:hover { background: #0284c7; }
            QPushButton#armButton { background: #166534; border-color: #4ade80; }
            QPushButton#armButton:hover { background: #15803d; }
            QPushButton#stopButton { background: #b91c1c; border-color: #f87171; }
            QPushButton#stopButton:hover { background: #dc2626; }
            QSlider::groove:horizontal {
                height: 10px; background: #475569; border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: #0ea5e9; border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #f8fafc; border: 2px solid #0284c7;
                width: 24px; margin: -8px 0; border-radius: 12px;
            }
            QSlider:disabled { opacity: 0.45; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("ODrive 电机手动控制")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #f8fafc;")
        subtitle = QLabel("AXIS0 · 速度模式 · 最大 ±5.0 圈/秒")
        subtitle.setStyleSheet("font-size: 13px; color: #94a3b8;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        connection_group = QGroupBox("1 · 连接驱动器")
        connection_layout = QHBoxLayout(connection_group)
        connection_layout.setSpacing(12)
        connection_layout.addWidget(QLabel("串口"))
        self.port_input = QLineEdit("COM32", self)
        self.port_input.setPlaceholderText("例如 COM32")
        connection_layout.addWidget(self.port_input, 1)
        self.connect_button = QPushButton("连接")
        self.connect_button.setObjectName("connectButton")
        self.connect_button.clicked.connect(self.connect_odrive)
        connection_layout.addWidget(self.connect_button)
        layout.addWidget(connection_group)

        state_group = QGroupBox("2 · 驱动状态")
        state_layout = QGridLayout(state_group)
        state_layout.setHorizontalSpacing(16)
        state_layout.setVerticalSpacing(12)
        state_layout.addWidget(QLabel("连接状态"), 0, 0)
        self.connection_status = QLabel("● 未连接")
        state_layout.addWidget(self.connection_status, 0, 1)
        state_layout.addWidget(QLabel("AXIS0 状态"), 1, 0)
        self.axis_status_label = QLabel("未知")
        state_layout.addWidget(self.axis_status_label, 1, 1)
        self.feedback_label = QLabel("请先连接 COM32；连接操作不会启动电机。")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.feedback_label.setStyleSheet(
            "background: #0f172a; border-left: 4px solid #64748b; "
            "border-radius: 5px; padding: 12px; color: #cbd5e1;"
        )
        state_layout.addWidget(self.feedback_label, 2, 0, 1, 3)
        layout.addWidget(state_group)

        control_group = QGroupBox("3 · 使能与速度控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(14)
        action_layout = QHBoxLayout()
        self.clear_errors_button = QPushButton("清除错误")
        self.clear_errors_button.clicked.connect(self.clear_errors)
        self.arm_button = QPushButton("使能 AXIS0")
        self.arm_button.setObjectName("armButton")
        self.arm_button.clicked.connect(self.arm_axis)
        self.stop_button = QPushButton("停止并进入 IDLE")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_axis)
        action_layout.addWidget(self.clear_errors_button)
        action_layout.addWidget(self.arm_button)
        action_layout.addWidget(self.stop_button)
        control_layout.addLayout(action_layout)

        self.velocity_label = QLabel("0.0 圈/秒")
        self.velocity_label.setAlignment(Qt.AlignCenter)
        self.velocity_label.setStyleSheet(
            "font-size: 32px; font-weight: 700; color: #7dd3fc; padding: 8px;"
        )
        control_layout.addWidget(self.velocity_label)
        self.velocity_slider = QSlider(Qt.Horizontal, self)
        self.velocity_slider.setRange(
            int(-MAX_SPEED_REV_S * 10), int(MAX_SPEED_REV_S * 10)
        )
        self.velocity_slider.setValue(0)
        self.velocity_slider.setTickPosition(QSlider.TicksBelow)
        self.velocity_slider.setTickInterval(10)
        self.velocity_slider.valueChanged.connect(self.slider_velocity_changed)
        control_layout.addWidget(self.velocity_slider)
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("反转  -5.0"))
        scale_layout.addStretch()
        scale_layout.addWidget(QLabel("停止  0"))
        scale_layout.addStretch()
        scale_layout.addWidget(QLabel("正转  +5.0"))
        control_layout.addLayout(scale_layout)
        layout.addWidget(control_group)

        safety = QLabel("⚠ 安全提示：先架空车轮并准备急停。停止时点击红色按钮，不要直接拔串口。")
        safety.setWordWrap(True)
        safety.setStyleSheet(
            "background: #422006; border: 1px solid #a16207; border-radius: 7px; "
            "padding: 12px; color: #fde68a; font-weight: 600;"
        )
        layout.addWidget(safety)
        layout.addStretch()
        self.set_controls_enabled(False)
        self.update_state_display("disconnected")

    def update_state_display(self, state, detail=None):
        states = {
            "disconnected": ("● 未连接", "#94a3b8", "未连接", "#94a3b8"),
            "connected": ("● 已连接", "#38bdf8", "IDLE / 未使能", "#fbbf24"),
            "armed": ("● 已连接", "#4ade80", "CLOSED LOOP / 已使能", "#4ade80"),
            "error": ("● 连接异常", "#f87171", "错误", "#f87171"),
        }
        connection_text, connection_color, axis_text, axis_color = states[state]
        self.connection_status.setText(connection_text)
        self.connection_status.setStyleSheet(
            f"color: {connection_color}; font-weight: 700; font-size: 14px;"
        )
        self.axis_status_label.setText(axis_text)
        self.axis_status_label.setStyleSheet(
            f"color: {axis_color}; font-weight: 700; font-size: 14px;"
        )
        if detail is not None:
            self.feedback_label.setText(detail)
            self.feedback_label.setStyleSheet(
                f"background: #0f172a; border-left: 4px solid {axis_color}; "
                "border-radius: 5px; padding: 12px; color: #e2e8f0;"
            )

    def set_controls_enabled(self, connected):
        self.clear_errors_button.setEnabled(connected)
        self.arm_button.setEnabled(connected)
        self.stop_button.setEnabled(connected)
        self.velocity_slider.setEnabled(connected and self.armed)

    def read_int(self, property_name):
        response = self.odrive.read_parameter(property_name)
        if not response:
            raise RuntimeError(f"No response for {property_name}")
        return int(float(response))

    def axis_status(self):
        return {
            "state": self.read_int("axis0.current_state"),
            "axis_error": self.read_int("axis0.error"),
            "motor_error": self.read_int("axis0.motor.error"),
            "encoder_error": self.read_int("axis0.encoder.error"),
            "motor_calibrated": self.read_int("axis0.motor.is_calibrated"),
            "encoder_ready": self.read_int("axis0.encoder.is_ready"),
            "control_mode": self.read_int(
                "axis0.controller.config.control_mode"
            ),
            "input_mode": self.read_int("axis0.controller.config.input_mode"),
        }

    @staticmethod
    def error_summary(status):
        return (
            f"state={status['state']}, axis_error={status['axis_error']}, "
            f"motor_error={status['motor_error']}, "
            f"encoder_error={status['encoder_error']}, "
            f"control_mode={status['control_mode']}, "
            f"input_mode={status['input_mode']}"
        )

    def connect_odrive(self):
        port = self.port_input.text().strip()
        if not port:
            self.feedback_label.setText("Enter a serial port")
            return
        self.close_odrive(show_status=False)
        try:
            drive = ODriveAsciiProtocol(port, BAUDRATE)
            self.odrive = drive
            status = self.axis_status()
            vbus = float(self.odrive.read_parameter("vbus_voltage"))
            detail = (
                f"已连接 {port}，波特率 {BAUDRATE}，母线电压 {vbus:.2f} V\n"
                f"诊断详情：{self.error_summary(status)}"
            )
            if any(
                status[name]
                for name in ("axis_error", "motor_error", "encoder_error")
            ):
                self.update_state_display("error", detail)
            else:
                self.update_state_display("connected", detail)
            self.set_controls_enabled(True)
        except Exception as error:
            self.close_odrive(show_status=False)
            self.update_state_display("error", f"连接失败：{error}")

    def clear_errors(self):
        if not self.odrive:
            return
        try:
            self.odrive.system_command("sc")
            time.sleep(0.2)
            status = self.axis_status()
            detail = f"清错结果：{self.error_summary(status)}"
            if any(
                status[name]
                for name in ("axis_error", "motor_error", "encoder_error")
            ):
                self.update_state_display("error", detail)
            else:
                self.update_state_display("connected", detail)
        except Exception as error:
            self.update_state_display("error", f"清除错误失败：{error}")

    def arm_axis(self):
        if not self.odrive:
            return
        try:
            status = self.axis_status()
            errors = (
                status["axis_error"],
                status["motor_error"],
                status["encoder_error"],
            )
            if any(errors):
                raise RuntimeError(
                    "AXIS0 still has errors; fix the cause and press Clear Errors first. "
                    + self.error_summary(status)
                )
            if not status["motor_calibrated"]:
                raise RuntimeError("AXIS0 motor is not calibrated")
            if not status["encoder_ready"]:
                raise RuntimeError("AXIS0 encoder is not ready")

            self.velocity_slider.blockSignals(True)
            self.velocity_slider.setValue(0)
            self.velocity_slider.blockSignals(False)
            self.odrive.motor_velocity(AXIS, 0.0)
            self.odrive.write_parameter("axis0.controller.config.input_mode", 1)
            self.odrive.write_parameter("axis0.controller.config.control_mode", 2)
            self.odrive.write_parameter("axis0.requested_state", 8)
            time.sleep(0.3)
            status = self.axis_status()
            if (
                status["state"] != 8
                or status["control_mode"] != 2
                or status["input_mode"] != 1
                or any(
                    status[name]
                    for name in ("axis_error", "motor_error", "encoder_error")
                )
            ):
                raise RuntimeError(
                    "AXIS0 did not enter velocity closed loop: "
                    + self.error_summary(status)
                )
            self.armed = True
            self.set_controls_enabled(True)
            self.update_state_display(
                "armed", "AXIS0 已进入速度闭环，当前目标速度为 0.0 圈/秒。"
            )
        except Exception as error:
            self.armed = False
            self.set_controls_enabled(self.odrive is not None)
            self.update_state_display("error", f"使能失败：{error}")

    def slider_velocity_changed(self, value):
        velocity = value / 10.0
        self.velocity_label.setText(f"{velocity:+.1f} 圈/秒")
        if not self.odrive or not self.armed:
            return
        try:
            self.odrive.motor_velocity(AXIS, velocity)
            direction = "正转" if velocity > 0 else "反转" if velocity < 0 else "停止"
            self.update_state_display(
                "armed", f"速度指令已发送：{direction} {abs(velocity):.1f} 圈/秒"
            )
        except Exception as error:
            self.update_state_display("error", f"速度指令发送失败：{error}")
            self.stop_axis()

    def stop_axis(self):
        if not self.odrive:
            return
        self.velocity_slider.blockSignals(True)
        self.velocity_slider.setValue(0)
        self.velocity_slider.blockSignals(False)
        self.velocity_label.setText("0.0 圈/秒")
        try:
            self.odrive.motor_velocity(AXIS, 0.0)
            time.sleep(0.1)
            self.odrive.write_parameter("axis0.requested_state", 1)
            self.armed = False
            self.set_controls_enabled(True)
            status = self.axis_status()
            self.update_state_display(
                "connected", "已发送零速度并进入 IDLE。\n" + self.error_summary(status)
            )
        except Exception as error:
            self.armed = False
            self.set_controls_enabled(self.odrive is not None)
            self.update_state_display(
                "error", f"停止失败，请立即使用急停：{error}"
            )

    def close_odrive(self, show_status=True):
        if self.odrive and self.armed:
            self.stop_axis()
        if self.odrive:
            self.odrive.close()
            self.odrive = None
        self.armed = False
        self.set_controls_enabled(False)
        if show_status:
            self.update_state_display("disconnected", "串口连接已关闭。")

    def closeEvent(self, event):
        self.close_odrive(show_status=False)
        event.accept()


def main():
    app = QApplication(sys.argv)
    motor_control_ui = MotorControlUI()
    motor_control_ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
