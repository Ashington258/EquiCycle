# motor_control_ui.py
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QSlider,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


class MotorControlUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.odrive = None

    def init_ui(self):
        self.setWindowTitle("Motor Control")
        self.resize(400, 400)

        # Dark Theme Palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(dark_palette)

        # Layout
        layout = QVBoxLayout()

        # Connection Controls
        connection_layout = QHBoxLayout()
        self.port_input = QLineEdit(self)
        self.port_input.setPlaceholderText("Enter port (e.g., COM3 or /dev/ttyACM0)")
        self.port_input.setStyleSheet("background-color: #353535; color: white;")
        connection_layout.addWidget(self.port_input)

        connect_button = QPushButton("Connect", self)
        connect_button.clicked.connect(self.connect_odrive)
        connection_layout.addWidget(connect_button)

        layout.addLayout(connection_layout)

        # Motor Position Controls
        position_layout = QVBoxLayout()
        self.position_slider = QSlider(Qt.Horizontal, self)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(1000)  # Example range for motor position
        self.position_slider.setValue(000)
        self.position_slider.valueChanged.connect(self.slider_position_changed)
        position_layout.addWidget(QLabel("Motor Position (turns):"))
        position_layout.addWidget(self.position_slider)

        layout.addLayout(position_layout)

        # Motor Velocity Controls
        velocity_layout = QVBoxLayout()
        self.velocity_slider = QSlider(Qt.Horizontal, self)
        self.velocity_slider.setMinimum(-200)
        self.velocity_slider.setMaximum(200)  # Example range for motor velocity
        self.velocity_slider.setValue(0)
        self.velocity_slider.valueChanged.connect(self.slider_velocity_changed)
        velocity_layout.addWidget(QLabel("Motor Velocity (turns/s):"))
        velocity_layout.addWidget(self.velocity_slider)

        layout.addLayout(velocity_layout)

        # Feedback Display
        self.feedback_label = QLabel("Motor Feedback: Not Connected", self)
        layout.addWidget(self.feedback_label)

        # Update Watchdog Button
        update_watchdog_button = QPushButton("Update Watchdog", self)
        update_watchdog_button.clicked.connect(self.update_watchdog)
        layout.addWidget(update_watchdog_button)

        # Close Connection Button
        close_button = QPushButton("Close Connection", self)
        close_button.clicked.connect(self.close_odrive)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def connect_odrive(self):
        port = self.port_input.text()
        if port:
            try:
                self.odrive = ODriveAsciiProtocol(port)
                self.feedback_label.setText(f"Connected to ODrive on {port}")
            except Exception as e:
                self.feedback_label.setText(f"Failed to connect: {e}")

    def slider_position_changed(self, value):
        if self.odrive:
            position = value / 100.0  # Convert to a float range if necessary
            self.odrive.motor_position(0, position)
            self.feedback_label.setText(f"Motor position set to {position}")
        else:
            self.feedback_label.setText("ODrive not connected")

    def slider_velocity_changed(self, value):
        if self.odrive:
            velocity = value / 10.0  # Convert to a suitable velocity range
            self.odrive.motor_velocity(0, velocity)
            self.feedback_label.setText(f"Motor velocity set to {velocity}")
        else:
            self.feedback_label.setText("ODrive not connected")

    def update_watchdog(self):
        if self.odrive:
            self.odrive.update_watchdog(0)
            self.feedback_label.setText("Watchdog updated")
        else:
            self.feedback_label.setText("ODrive not connected")

    def close_odrive(self):
        if self.odrive:
            self.odrive.close()
            self.odrive = None
            self.feedback_label.setText("Connection closed")


def main():
    app = QApplication(sys.argv)
    motor_control_ui = MotorControlUI()
    motor_control_ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
