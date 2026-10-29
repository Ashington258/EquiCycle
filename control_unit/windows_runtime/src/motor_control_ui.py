# motor_control_ui.py
import math
import os
import sys
import tempfile
import time
from collections import deque
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QSlider,
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
)
from PyQt5.QtCore import QLockFile, Qt, QTimer
from PyQt5.QtGui import QPalette, QColor
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


class MotorControlUI(QWidget):
    VELOCITY_LIMIT_REV_S = 35.0
    DEFAULT_CURRENT_LIMIT_A = 30.0
    DEFAULT_TORQUE_LIMIT_NM = 1.0
    DEFAULT_VELOCITY_RAMP_RATE_REV_S2 = 60.0
    MAX_CURRENT_LIMIT_A = 150.0
    MAX_TORQUE_LIMIT_NM = 40.0
    MAX_VELOCITY_RAMP_RATE_REV_S2 = 400.0
    STOP_SPEED_THRESHOLD_REV_S = 0.05
    STOP_STABLE_TIME_S = 0.20
    STOP_TIMEOUT_S = 8.0

    def __init__(self):
        super().__init__()
        self.odrive = None
        self.current_motor = 0
        self.configured_control_mode = None
        self.torque_constant = None
        self.torque_limit = 0.0
        self._stored_limits = None
        self._velocity_samples = deque(maxlen=5)
        self._peak_velocity_rate = 0.0
        self._peak_current = 0.0
        self.init_ui()

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
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(150, 156, 166))
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(150, 156, 166))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(150, 156, 166))
        QApplication.setPalette(dark_palette)
        self.setStyleSheet(
            "QWidget { background-color: #25282d; color: #f2f4f7; }"
            "QLabel { background-color: transparent; color: #f2f4f7; }"
            "QLineEdit, QComboBox, QDoubleSpinBox {"
            " background-color: #111318; color: #ffffff; border: 1px solid #7c8594;"
            " border-radius: 4px; padding: 5px; selection-background-color: #2878c7;"
            "}"
            "QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled {"
            " background-color: #30343b; color: #aeb4bf; border-color: #555d69;"
            "}"
            "QPushButton { background-color: #3b82c4; color: #ffffff;"
            " border: 1px solid #72b4ed; border-radius: 4px; padding: 7px 10px; }"
            "QPushButton:hover { background-color: #4b94d4; }"
            "QPushButton:pressed { background-color: #28699f; }"
            "QPushButton:disabled { background-color: #3b4048; color: #9ba2ad;"
            " border-color: #555d69; }"
            "QSlider::groove:horizontal { height: 6px; background: #111318;"
            " border: 1px solid #687180; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #62b0ee; border: 1px solid #d7ecff;"
            " width: 16px; margin: -6px 0; border-radius: 8px; }"
        )

        # Layout
        layout = QVBoxLayout()

        # Connection Controls
        connection_layout = QHBoxLayout()
        self.port_input = QLineEdit(self)
        self.port_input.setPlaceholderText("Enter port (e.g., COM3 or /dev/ttyACM0)")
        connection_layout.addWidget(self.port_input)

        self.connect_button = QPushButton("Connect", self)
        self.connect_button.clicked.connect(self.connect_odrive)
        connection_layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton("Disconnect", self)
        self.disconnect_button.clicked.connect(self.disconnect_odrive)
        self.disconnect_button.setEnabled(False)
        connection_layout.addWidget(self.disconnect_button)

        layout.addLayout(connection_layout)

        # Motor Selection
        motor_selection_layout = QHBoxLayout()
        self.motor_selector = QComboBox(self)
        self.motor_selector.addItems(["Motor 0", "Motor 1"])
        self.motor_selector.currentIndexChanged.connect(self.change_motor)
        motor_selection_layout.addWidget(QLabel("Select Motor:"))
        motor_selection_layout.addWidget(self.motor_selector)

        layout.addLayout(motor_selection_layout)

        # Motor Position Controls
        position_layout = QVBoxLayout()
        self.position_slider = QSlider(Qt.Horizontal, self)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(1000)  # Example range for motor position
        self.position_slider.setValue(0)
        self.position_slider.valueChanged.connect(self.slider_position_changed)
        position_layout.addWidget(QLabel("Motor Position (turns):"))
        position_layout.addWidget(self.position_slider)

        layout.addLayout(position_layout)

        # Motor Velocity Controls
        velocity_layout = QVBoxLayout()
        self.velocity_slider = QSlider(Qt.Horizontal, self)
        self.velocity_slider.setMinimum(int(-self.VELOCITY_LIMIT_REV_S * 10))
        self.velocity_slider.setMaximum(int(self.VELOCITY_LIMIT_REV_S * 10))
        self.velocity_slider.setValue(0)
        self.velocity_slider.valueChanged.connect(self.slider_velocity_changed)
        velocity_layout.addWidget(QLabel("Motor Velocity (turns/s):"))
        velocity_layout.addWidget(self.velocity_slider)

        layout.addLayout(velocity_layout)

        limits_layout = QHBoxLayout()
        limits_layout.addWidget(QLabel("Current limit:"))
        self.current_limit_input = QDoubleSpinBox(self)
        self.current_limit_input.setRange(0.0, self.MAX_CURRENT_LIMIT_A)
        self.current_limit_input.setDecimals(1)
        self.current_limit_input.setSingleStep(1.0)
        self.current_limit_input.setSuffix(" A")
        self.current_limit_input.setValue(self.DEFAULT_CURRENT_LIMIT_A)
        self.current_limit_input.valueChanged.connect(self._current_limit_changed)
        limits_layout.addWidget(self.current_limit_input)

        limits_layout.addWidget(QLabel("Torque limit:"))
        self.torque_limit_input = QDoubleSpinBox(self)
        self.torque_limit_input.setRange(0.0, self.MAX_TORQUE_LIMIT_NM)
        self.torque_limit_input.setDecimals(2)
        self.torque_limit_input.setSingleStep(0.1)
        self.torque_limit_input.setSuffix(" Nm")
        self.torque_limit_input.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.torque_limit_input.setValue(self.DEFAULT_TORQUE_LIMIT_NM)
        self.torque_limit_input.valueChanged.connect(self._torque_limit_changed)
        limits_layout.addWidget(self.torque_limit_input)

        limits_layout.addWidget(QLabel("Velocity change limit:"))
        self.velocity_ramp_rate_input = QDoubleSpinBox(self)
        self.velocity_ramp_rate_input.setRange(
            0.1, self.MAX_VELOCITY_RAMP_RATE_REV_S2
        )
        self.velocity_ramp_rate_input.setDecimals(1)
        self.velocity_ramp_rate_input.setSingleStep(10.0)
        self.velocity_ramp_rate_input.setSuffix(" rev/s²")
        self.velocity_ramp_rate_input.setValue(
            self.DEFAULT_VELOCITY_RAMP_RATE_REV_S2
        )
        self.velocity_ramp_rate_input.valueChanged.connect(self._update_limit_info)
        limits_layout.addWidget(self.velocity_ramp_rate_input)

        self.apply_limits_button = QPushButton("Apply Limits", self)
        self.apply_limits_button.clicked.connect(self.apply_motor_limits)
        self.apply_limits_button.setEnabled(False)
        limits_layout.addWidget(self.apply_limits_button)
        layout.addLayout(limits_layout)

        self.limit_info_label = QLabel(
            "Current, torque, and velocity-change values are independent ODrive limits. "
            "They are ceilings, not output commands; actual acceleration can be lower "
            "under load.",
            self,
        )
        self.limit_info_label.setWordWrap(True)
        layout.addWidget(self.limit_info_label)
        self.stored_limit_label = QLabel("ODrive limits: not read", self)
        self.stored_limit_label.setWordWrap(True)
        layout.addWidget(self.stored_limit_label)

        # Feedback Display
        self.feedback_label = QLabel("Motor Feedback: Not Connected", self)
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.feedback_label)

        self.velocity_rate_label = QLabel(
            "Actual speed change rate: -- rev/s² | peak: 0.00", self
        )
        layout.addWidget(self.velocity_rate_label)
        self.current_feedback_label = QLabel("Motor current: -- A | peak: 0.00 A", self)
        layout.addWidget(self.current_feedback_label)
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.setInterval(20)
        self.telemetry_timer.timeout.connect(self.update_live_velocity_rate)
        self.telemetry_timer.start()

        self.reset_rate_peak_button = QPushButton("Reset Rate Peak", self)
        self.reset_rate_peak_button.clicked.connect(self.reset_velocity_rate_peak)
        layout.addWidget(self.reset_rate_peak_button)

        most_speed_test_layout = QVBoxLayout()
        self.most_speed_test_label = QLabel(
            f"most_speed+test: {self.VELOCITY_LIMIT_REV_S:.1f} rev/s "
            f"(-{self.VELOCITY_LIMIT_REV_S:.1f} → +{self.VELOCITY_LIMIT_REV_S:.1f} "
            f"→ -{self.VELOCITY_LIMIT_REV_S:.1f})",
            self,
        )
        most_speed_test_layout.addWidget(self.most_speed_test_label)
        self.most_speed_test_slider = QSlider(Qt.Horizontal, self)
        self.most_speed_test_slider.setRange(10, int(self.VELOCITY_LIMIT_REV_S * 10))
        self.most_speed_test_slider.setValue(int(self.VELOCITY_LIMIT_REV_S * 10))
        self.most_speed_test_slider.valueChanged.connect(
            self.update_most_speed_test_label
        )
        most_speed_test_layout.addWidget(self.most_speed_test_slider)
        layout.addLayout(most_speed_test_layout)

        self.max_rate_test_button = QPushButton("most_speed+test", self)
        self.max_rate_test_button.clicked.connect(self.test_max_velocity_rate)
        self.max_rate_test_button.setEnabled(False)
        layout.addWidget(self.max_rate_test_button)

        # Update Watchdog Button
        update_watchdog_button = QPushButton("Update Watchdog", self)
        update_watchdog_button.clicked.connect(self.update_watchdog)
        layout.addWidget(update_watchdog_button)

        self.read_status_button = QPushButton("Read Motor Status", self)
        self.read_status_button.clicked.connect(self.read_motor_status)
        self.read_status_button.setEnabled(False)
        layout.addWidget(self.read_status_button)

        self.clear_errors_button = QPushButton(
            "Clear Errors && Enter Closed Loop", self
        )
        self.clear_errors_button.clicked.connect(self.clear_odrive_errors)
        self.clear_errors_button.setEnabled(False)
        layout.addWidget(self.clear_errors_button)

        self.setLayout(layout)

    def connect_odrive(self):
        if self.odrive is not None:
            return

        port = self.port_input.text().strip()
        if not port:
            self.feedback_label.setText("Enter a serial port before connecting")
            return

        odrive = None
        try:
            odrive = ODriveAsciiProtocol(port)
            vbus_voltage = float(odrive.read_parameter("vbus_voltage"))
            axis = self.current_motor
            current_limit = float(
                odrive.read_parameter(f"axis{axis}.motor.config.current_lim")
            )
            torque_constant = float(
                odrive.read_parameter(f"axis{axis}.motor.config.torque_constant")
            )
            if not math.isfinite(torque_constant) or torque_constant <= 0:
                raise RuntimeError(
                    f"invalid motor torque constant: {torque_constant!r}"
                )
            torque_limit = float(
                odrive.read_parameter(f"axis{axis}.motor.config.torque_lim")
            )
            ramp_rate = float(
                odrive.read_parameter(f"axis{axis}.controller.config.vel_ramp_rate")
            )
        except Exception as exc:
            if odrive is not None:
                try:
                    odrive.close()
                except Exception:
                    pass
            error_message = self._operation_error(f"connect to {port}", exc)
            self.feedback_label.setText(error_message)
            QMessageBox.critical(self, "ODrive connection failed", error_message)
            self._set_connected_state(False)
            return

        self.odrive = odrive
        self.reset_velocity_rate_peak()
        self.configured_control_mode = None
        self.torque_constant = torque_constant
        self.current_limit_input.setValue(
            min(current_limit, self.MAX_CURRENT_LIMIT_A)
        )
        self._set_torque_limit_value(torque_limit)
        self.velocity_ramp_rate_input.setValue(
            min(ramp_rate, self.MAX_VELOCITY_RAMP_RATE_REV_S2)
        )
        self._set_connected_state(True)
        self._show_stored_limits(current_limit, torque_limit, ramp_rate)
        self._update_limit_info()
        self.feedback_label.setText(
            f"Connected to ODrive on {port} | vbus={vbus_voltage:.2f} V | "
            f"stored torque limit={torque_limit:.2f} Nm"
        )

    def _operation_error(self, operation, exc):
        message = f"Failed to {operation}: {type(exc).__name__}: {exc}"
        if "winerror 31" in str(exc).lower() or "none, 31" in str(exc).lower():
            message += (
                "\nWindows could not configure the ODrive USB serial device. "
                "Unplug the ODrive USB cable for 5 seconds, reconnect it, then retry."
            )
        return message

    def _run_command(self, operation, command):
        if self.odrive is None:
            self.feedback_label.setText("ODrive not connected")
            return False
        try:
            command()
        except Exception as exc:
            error_message = self._operation_error(operation, exc)
            self.feedback_label.setText(error_message)
            QMessageBox.critical(self, "ODrive command failed", error_message)
            return False
        return True

    def _set_control_mode(self, control_mode):
        axis = self.current_motor
        self.odrive.write_parameter(
            f"axis{axis}.controller.config.vel_limit", self.VELOCITY_LIMIT_REV_S
        )
        input_mode = 2 if control_mode == 2 else 1
        self.odrive.write_parameter(
            f"axis{axis}.controller.config.input_mode", input_mode
        )
        self.odrive.write_parameter(
            f"axis{axis}.controller.config.control_mode", control_mode
        )
        self.configured_control_mode = control_mode
        if self.__dict__.get("_stored_limits") is not None:
            self._show_stored_limits(*self._stored_limits[:3], input_mode)

    def _set_connected_state(self, connected):
        self.port_input.setEnabled(not connected)
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
        self.read_status_button.setEnabled(connected)
        self.clear_errors_button.setEnabled(connected)
        self.apply_limits_button.setEnabled(connected)
        self.max_rate_test_button.setEnabled(connected)

    def change_motor(self, index):
        self.current_motor = index
        self.reset_velocity_rate_peak()
        self.configured_control_mode = None
        self.torque_constant = None
        if self.odrive is None:
            self._update_limit_info()
            return

        def read_axis_limits():
            axis = self.current_motor
            current_limit = float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.current_lim")
            )
            self.torque_constant = float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.torque_constant")
            )
            if not math.isfinite(self.torque_constant) or self.torque_constant <= 0:
                raise RuntimeError(
                    f"invalid motor torque constant: {self.torque_constant!r}"
                )
            stored_torque_limit = float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.torque_lim")
            )
            ramp_rate = float(
                self.odrive.read_parameter(
                    f"axis{axis}.controller.config.vel_ramp_rate"
                )
            )
            self.current_limit_input.setValue(min(current_limit, self.MAX_CURRENT_LIMIT_A))
            self._set_torque_limit_value(stored_torque_limit)
            self.velocity_ramp_rate_input.setValue(
                min(ramp_rate, self.MAX_VELOCITY_RAMP_RATE_REV_S2)
            )
            self._show_stored_limits(current_limit, stored_torque_limit, ramp_rate)

        if self._run_command("read motor limits", read_axis_limits):
            self._update_limit_info()

    def update_most_speed_test_label(self, value):
        speed = value / 10.0
        self.most_speed_test_label.setText(
            f"most_speed+test: {speed:.1f} rev/s "
            f"(-{speed:.1f} → +{speed:.1f} → -{speed:.1f})"
        )

    def _show_stored_limits(self, current, torque, ramp, input_mode=None):
        self._stored_limits = (current, torque, ramp, input_mode)
        ramp_status = (
            "ramp mode selected" if input_mode == 2 else
            "ramp mode not selected" if input_mode is not None else "ramp mode not checked"
        )
        self.stored_limit_label.setText(
            f"ODrive limits (last read): current={current:.1f} A, "
            f"torque={torque:.2f} Nm, velocity ramp={ramp:.1f} rev/s²; "
            f"{ramp_status}. Controls above are pending until Apply Limits."
        )

    def _update_limit_info(self):
        self.limit_info_label.setText(
            f"Current limit: {self.current_limit_input.value():.1f} A | "
            f"torque limit: {self.torque_limit_input.value():.2f} Nm | "
            f"velocity change limit: "
            f"{self.velocity_ramp_rate_input.value():.1f} rev/s² | "
            f"speed limit: {self.VELOCITY_LIMIT_REV_S:.1f} rev/s"
        )

    def _set_torque_limit_value(self, torque_limit):
        if not math.isfinite(torque_limit) or torque_limit < 0:
            raise RuntimeError(f"invalid torque limit: {torque_limit!r}")
        displayed_limit = min(torque_limit, self.MAX_TORQUE_LIMIT_NM)
        self.torque_limit = torque_limit
        self.torque_limit_input.setValue(displayed_limit)
        return displayed_limit

    def _current_limit_changed(self):
        self._update_limit_info()

    def _torque_limit_changed(self, value):
        self._update_limit_info()

    def apply_motor_limits(self):
        axis = self.current_motor
        current_limit = self.current_limit_input.value()
        torque_limit = self.torque_limit_input.value()
        ramp_rate = self.velocity_ramp_rate_input.value()

        def apply_limits():
            self._stop_motor(axis)
            self.odrive.write_parameter(
                f"axis{axis}.motor.config.current_lim", current_limit
            )
            actual_limit = float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.current_lim")
            )
            if abs(actual_limit - current_limit) > 0.1:
                raise RuntimeError(
                    f"current limit readback mismatch: requested={current_limit:.1f}, "
                    f"actual={actual_limit:.1f}"
                )
            self.odrive.write_parameter(
                f"axis{axis}.motor.config.torque_lim", torque_limit
            )
            actual_torque_limit = float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.torque_lim")
            )
            if abs(actual_torque_limit - torque_limit) > 0.01:
                raise RuntimeError(
                    "torque limit readback mismatch: "
                    f"requested={torque_limit:.2f}, actual={actual_torque_limit:.2f}"
                )
            self.torque_limit = torque_limit
            self.odrive.write_parameter(
                f"axis{axis}.controller.config.vel_ramp_rate", ramp_rate
            )
            actual_ramp_rate = float(
                self.odrive.read_parameter(
                    f"axis{axis}.controller.config.vel_ramp_rate"
                )
            )
            if abs(actual_ramp_rate - ramp_rate) > 0.1:
                raise RuntimeError(
                    "velocity change limit readback mismatch: "
                    f"requested={ramp_rate:.1f}, actual={actual_ramp_rate:.1f}"
                )
            self._set_control_mode(2)
            actual_velocity_limit = float(
                self.odrive.read_parameter(f"axis{axis}.controller.config.vel_limit")
            )
            if abs(actual_velocity_limit - self.VELOCITY_LIMIT_REV_S) > 0.1:
                raise RuntimeError(
                    "velocity limit readback mismatch: "
                    f"requested={self.VELOCITY_LIMIT_REV_S:.1f}, "
                    f"actual={actual_velocity_limit:.1f}"
                )
            actual_input_mode = self._read_int(
                f"axis{axis}.controller.config.input_mode"
            )
            if actual_input_mode != 2:
                raise RuntimeError(
                    "velocity ramp mode readback mismatch: "
                    f"requested=2, actual={actual_input_mode}"
                )
            actual_control_mode = self._read_int(
                f"axis{axis}.controller.config.control_mode"
            )
            if actual_control_mode != 2:
                raise RuntimeError(
                    "velocity control mode readback mismatch: "
                    f"requested=2, actual={actual_control_mode}"
                )
            self.odrive.motor_velocity(axis, 0, 0)
            self.odrive.write_parameter(f"axis{axis}.requested_state", 8)
            time.sleep(0.3)
            if self._read_int(f"axis{axis}.current_state") != 8:
                raise RuntimeError("motor did not return to closed loop")

        if self._run_command("apply motor limits", apply_limits):
            self._show_stored_limits(current_limit, torque_limit, ramp_rate, 2)
            self._update_limit_info()
            self.feedback_label.setText(
                f"Motor {axis} limits applied: current={current_limit:.1f} A, "
                f"torque={torque_limit:.2f} Nm, "
                f"velocity limit={self.VELOCITY_LIMIT_REV_S:.1f} rev/s, "
                f"velocity change limit={ramp_rate:.1f} rev/s²"
            )

    def _measure_velocity_transition(
        self, target_velocity, timeout=2.0, update_peak_display=False
    ):
        start_time = time.monotonic()
        start_velocity = float(
            self.odrive.request_feedback(self.current_motor).split()[1]
        )
        samples = [(start_time, start_velocity)]
        peak_current = 0.0
        peak_setpoint_current = 0.0
        final_input_velocity = start_velocity
        self.odrive.motor_velocity(self.current_motor, target_velocity)
        reached_time = None
        while time.monotonic() - start_time < timeout:
            now = time.monotonic()
            velocity = float(
                self.odrive.request_feedback(self.current_motor).split()[1]
            )
            measured_current = abs(
                float(
                    self.odrive.read_parameter(
                        f"axis{self.current_motor}.motor.current_control.Iq_measured"
                    )
                )
            )
            setpoint_current = abs(
                float(
                    self.odrive.read_parameter(
                        f"axis{self.current_motor}.motor.current_control.Iq_setpoint"
                    )
                )
            )
            final_input_velocity = float(
                self.odrive.read_parameter(
                    f"axis{self.current_motor}.controller.input_vel"
                )
            )
            peak_current = max(peak_current, measured_current)
            peak_setpoint_current = max(peak_setpoint_current, setpoint_current)
            self._peak_current = max(self._peak_current, measured_current)
            self.current_feedback_label.setText(
                f"Motor current: requested={setpoint_current:.2f} A | "
                f"measured={measured_current:.2f} A | "
                f"measured peak={self._peak_current:.2f} A"
            )
            samples.append((now, velocity))
            if update_peak_display:
                rate = self._record_velocity_sample(now, velocity)
                rate_text = "--" if rate is None else f"{rate:+.2f}"
                self.velocity_rate_label.setText(
                    f"Actual speed change rate: {rate_text} rev/s² | "
                    f"peak: {self._peak_velocity_rate:.2f} | "
                    f"speed: {velocity:.3f} rev/s"
                )
                QApplication.processEvents()
            if abs(velocity - target_velocity) <= 1.0:
                reached_time = now - start_time
                break
        rates = []
        window = deque(maxlen=5)
        for timestamp, velocity in samples:
            window.append((timestamp, velocity))
            if len(window) < 2:
                continue
            mean_time = sum(item[0] for item in window) / len(window)
            mean_velocity = sum(item[1] for item in window) / len(window)
            denominator = sum((item[0] - mean_time) ** 2 for item in window)
            if denominator > 0:
                rates.append(
                    sum(
                        (item[0] - mean_time) * (item[1] - mean_velocity)
                        for item in window
                    )
                    / denominator
                )
        return {
            "start_velocity": start_velocity,
            "target_velocity": target_velocity,
            "peak_rate": max((abs(rate) for rate in rates), default=0.0),
            "transition_time": reached_time,
            "peak_current": peak_current,
            "peak_setpoint_current": peak_setpoint_current,
            "final_velocity": samples[-1][1],
            "final_input_velocity": final_input_velocity,
            "samples": len(samples),
        }

    @staticmethod
    def _require_transition_reached(stage, result):
        if result["transition_time"] is not None:
            return
        raise RuntimeError(
            f"{stage} timed out before reaching target: "
            f"target={result['target_velocity']:.2f} rev/s, "
            f"ODrive input_vel={result['final_input_velocity']:.2f} rev/s, "
            f"actual={result['final_velocity']:.2f} rev/s, "
            f"peak Iq_setpoint={result['peak_setpoint_current']:.2f} A, "
            f"peak Iq_measured={result['peak_current']:.2f} A. "
            "The configured current and velocity-change values are ceilings; "
            "this diagnostic shows whether the velocity controller requested "
            "enough current to reach them."
        )

    def test_max_velocity_rate(self):
        if self.odrive is None:
            self.feedback_label.setText("ODrive not connected")
            return
        confirmation = QMessageBox.question(
            self,
            "Confirm suspended-wheel test",
            "This test commands the selected negative maximum speed to the "
            "selected positive maximum speed and back while enforcing the current "
            "and velocity-change limits shown in the UI. Keep the wheel suspended "
            "and emergency power-off ready. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return

        axis = self.current_motor
        current_limit = self.current_limit_input.value()
        torque_limit = self.torque_limit_input.value()
        ramp_rate = self.velocity_ramp_rate_input.value()
        test_speed = min(
            self.most_speed_test_slider.value() / 10.0,
            self.VELOCITY_LIMIT_REV_S,
        )
        if current_limit <= 0 or ramp_rate <= 0:
            self.feedback_label.setText(
                "Positive current and velocity-change limits are required"
            )
            return

        self.telemetry_timer.stop()
        self.reset_velocity_rate_peak()
        saved = {}
        try:
            for name in (
                "motor.config.current_lim",
                "motor.config.torque_lim",
                "controller.config.vel_ramp_rate",
                "controller.config.vel_limit",
                "controller.config.input_mode",
                "controller.config.control_mode",
            ):
                saved[name] = float(self.odrive.read_parameter(f"axis{axis}.{name}"))

            self._stop_motor(axis)
            self.odrive.write_parameter(
                f"axis{axis}.motor.config.current_lim", current_limit
            )
            actual_current_limit = float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.current_lim")
            )
            if abs(actual_current_limit - current_limit) > 0.1:
                raise RuntimeError(
                    "current limit readback mismatch during test: "
                    f"requested={current_limit:.1f}, actual={actual_current_limit:.1f}"
                )
            self.odrive.write_parameter(
                f"axis{axis}.motor.config.torque_lim", torque_limit
            )
            actual_torque_limit = float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.torque_lim")
            )
            if abs(actual_torque_limit - torque_limit) > 0.01:
                raise RuntimeError(
                    "torque limit readback mismatch: "
                    f"requested={torque_limit:.2f}, actual={actual_torque_limit:.2f}"
                )
            self.torque_limit = torque_limit
            self.odrive.write_parameter(
                f"axis{axis}.controller.config.vel_ramp_rate", ramp_rate
            )
            actual_ramp_rate = float(
                self.odrive.read_parameter(
                    f"axis{axis}.controller.config.vel_ramp_rate"
                )
            )
            if abs(actual_ramp_rate - ramp_rate) > 0.1:
                raise RuntimeError(
                    "velocity-change limit readback mismatch during test: "
                    f"requested={ramp_rate:.1f}, actual={actual_ramp_rate:.1f}"
                )
            self.odrive.write_parameter(
                f"axis{axis}.controller.config.vel_limit", test_speed
            )
            self.odrive.write_parameter(f"axis{axis}.controller.config.input_mode", 2)
            self.odrive.write_parameter(f"axis{axis}.controller.config.control_mode", 2)
            actual_test_ramp_rate = float(
                self.odrive.read_parameter(
                    f"axis{axis}.controller.config.vel_ramp_rate"
                )
            )
            actual_test_input_mode = self._read_int(
                f"axis{axis}.controller.config.input_mode"
            )
            actual_test_control_mode = self._read_int(
                f"axis{axis}.controller.config.control_mode"
            )
            if abs(actual_test_ramp_rate - ramp_rate) > 0.1:
                raise RuntimeError(
                    "velocity-change limit changed before test: "
                    f"requested={ramp_rate:.1f}, actual={actual_test_ramp_rate:.1f}"
                )
            if actual_test_input_mode != 2 or actual_test_control_mode != 2:
                raise RuntimeError(
                    "velocity ramp control mode readback mismatch during test: "
                    f"input_mode={actual_test_input_mode}, "
                    f"control_mode={actual_test_control_mode}"
                )
            self.odrive.write_parameter(f"axis{axis}.requested_state", 8)
            time.sleep(0.3)
            actual_state = self._read_int(f"axis{axis}.current_state")
            if actual_state != 8:
                raise RuntimeError(
                    f"motor did not enter closed loop for test; state={actual_state}"
                )

            transition_timeout = 2 * test_speed / ramp_rate + 2.0
            initial_transition = self._measure_velocity_transition(
                -test_speed,
                timeout=transition_timeout,
                update_peak_display=True,
            )
            self._require_transition_reached(
                f"0 to -{test_speed:.1f} rev/s transition", initial_transition
            )
            negative_to_positive = self._measure_velocity_transition(
                test_speed,
                timeout=transition_timeout,
                update_peak_display=True,
            )
            self._require_transition_reached(
                f"-{test_speed:.1f} to +{test_speed:.1f} rev/s transition",
                negative_to_positive,
            )
            positive_to_negative = self._measure_velocity_transition(
                -test_speed,
                timeout=transition_timeout,
                update_peak_display=True,
            )
            self._require_transition_reached(
                f"+{test_speed:.1f} to -{test_speed:.1f} rev/s transition",
                positive_to_negative,
            )

            errors = {
                name: self._read_int(f"axis{axis}.{name}")
                for name in ("error", "motor.error", "encoder.error")
            }
            if any(errors.values()):
                raise RuntimeError(f"ODrive errors during test: {errors}")

            def format_time(result):
                value = result["transition_time"]
                return "timeout" if value is None else f"{value:.3f} s"

            self.feedback_label.setText(
                "Limited-rate test complete: "
                f"test speed=±{test_speed:.1f} rev/s, "
                f"current limit={current_limit:.1f} A, "
                f"torque limit={torque_limit:.2f} Nm, "
                f"velocity-change limit={ramp_rate:.1f} rev/s²; "
                f"-{test_speed:.1f}→ {test_speed:.1f} "
                f"peak={negative_to_positive['peak_rate']:.1f} rev/s², "
                f"Iq requested/measured="
                f"{negative_to_positive['peak_setpoint_current']:.1f}/"
                f"{negative_to_positive['peak_current']:.1f} A, "
                f"time={format_time(negative_to_positive)}; "
                f"+{test_speed:.1f}→ {test_speed:.1f} "
                f"peak={positive_to_negative['peak_rate']:.1f} rev/s², "
                f"Iq requested/measured="
                f"{positive_to_negative['peak_setpoint_current']:.1f}/"
                f"{positive_to_negative['peak_current']:.1f} A, "
                f"time={format_time(positive_to_negative)}."
            )
        except Exception as exc:
            error_message = self._operation_error("run max-rate test", exc)
            self.feedback_label.setText(error_message)
            QMessageBox.critical(self, "Max-rate test failed", error_message)
        finally:
            try:
                self._stop_motor(axis)
                for name, value in saved.items():
                    self.odrive.write_parameter(f"axis{axis}.{name}", value)
                self.configured_control_mode = None
            except Exception as restore_exc:
                self.feedback_label.setText(
                    f"Test stopped, but configuration restore failed: {restore_exc}"
                )
            self._velocity_samples.clear()
            self.telemetry_timer.start()

    def slider_position_changed(self, value):
        position = value / 100.0

        def set_position():
            self._set_control_mode(3)
            self.odrive.motor_position(
                self.current_motor,
                position,
                velocity_lim=self.VELOCITY_LIMIT_REV_S,
                torque_lim=min(self.torque_limit, self.torque_limit_input.value()),
            )

        if self._run_command("set motor position", set_position):
            self.feedback_label.setText(
                f"Motor {self.current_motor} position set to {position}"
            )

    def slider_velocity_changed(self, value):
        velocity = max(
            -self.VELOCITY_LIMIT_REV_S,
            min(value / 10.0, self.VELOCITY_LIMIT_REV_S),
        )

        def set_velocity():
            self._set_control_mode(2)
            self.odrive.motor_velocity(self.current_motor, velocity)

        if self._run_command("set motor velocity", set_velocity):
            self.feedback_label.setText(
                f"Motor {self.current_motor} velocity set to {velocity}"
            )

    def update_watchdog(self):
        if self._run_command(
            "update watchdog",
            lambda: self.odrive.update_watchdog(self.current_motor),
        ):
            self.feedback_label.setText(
                f"Watchdog updated for Motor {self.current_motor}"
            )

    def _read_int(self, property_name):
        return int(float(self.odrive.read_parameter(property_name)))

    def _wait_until_stopped(self, axis):
        deadline = time.monotonic() + self.STOP_TIMEOUT_S
        stopped_since = None
        velocity = 0.0
        while time.monotonic() < deadline:
            feedback = self.odrive.request_feedback(axis).split()
            if len(feedback) != 2:
                raise RuntimeError(f"invalid velocity feedback while stopping: {feedback!r}")
            velocity = float(feedback[1])
            errors = {
                name: self._read_int(f"axis{axis}.{name}")
                for name in ("error", "motor.error", "encoder.error")
            }
            if any(errors.values()):
                raise RuntimeError(f"ODrive errors while stopping: {errors}")
            now = time.monotonic()
            if abs(velocity) <= self.STOP_SPEED_THRESHOLD_REV_S:
                stopped_since = stopped_since or now
                if now - stopped_since >= self.STOP_STABLE_TIME_S:
                    return
            else:
                stopped_since = None
            time.sleep(0.05)
        raise RuntimeError(
            f"等待机械停稳超时，最后反馈速度={velocity:.3f} rev/s"
        )

    def _stop_motor(self, axis):
        self.odrive.motor_velocity(axis, 0, 0)
        self._wait_until_stopped(axis)
        self.odrive.write_parameter(f"axis{axis}.requested_state", 1)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            state = self._read_int(f"axis{axis}.current_state")
            if state == 1:
                errors = {
                    name: self._read_int(f"axis{axis}.{name}")
                    for name in ("error", "motor.error", "encoder.error")
                }
                if any(errors.values()):
                    raise RuntimeError(f"ODrive errors after stopping: {errors}")
                return
            time.sleep(0.05)
        raise RuntimeError("motor did not return to IDLE after stopping")

    def _read_status(self):
        axis = self.current_motor
        position, velocity = self.odrive.request_feedback(axis).split()
        return {
            "vbus_voltage": float(self.odrive.read_parameter("vbus_voltage")),
            "state": self._read_int(f"axis{axis}.current_state"),
            "axis_error": self._read_int(f"axis{axis}.error"),
            "motor_error": self._read_int(f"axis{axis}.motor.error"),
            "encoder_error": self._read_int(f"axis{axis}.encoder.error"),
            "calibrated": self._read_int(f"axis{axis}.motor.is_calibrated"),
            "encoder_ready": self._read_int(f"axis{axis}.encoder.is_ready"),
            "control_mode": self._read_int(
                f"axis{axis}.controller.config.control_mode"
            ),
            "input_mode": self._read_int(
                f"axis{axis}.controller.config.input_mode"
            ),
            "velocity_limit": float(
                self.odrive.read_parameter(f"axis{axis}.controller.config.vel_limit")
            ),
            "input_velocity": float(
                self.odrive.read_parameter(f"axis{axis}.controller.input_vel")
            ),
            "current_limit": float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.current_lim")
            ),
            "torque_limit": float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.torque_lim")
            ),
            "torque_constant": float(
                self.odrive.read_parameter(f"axis{axis}.motor.config.torque_constant")
            ),
            "velocity_ramp_rate": float(
                self.odrive.read_parameter(
                    f"axis{axis}.controller.config.vel_ramp_rate"
                )
            ),
            "iq_measured": float(
                self.odrive.read_parameter(f"axis{axis}.motor.current_control.Iq_measured")
            ),
            "iq_setpoint": float(
                self.odrive.read_parameter(f"axis{axis}.motor.current_control.Iq_setpoint")
            ),
            "position": float(position),
            "velocity": float(velocity),
        }

    def _format_status(self, status):
        ready = (
            status["state"] == 8
            and status["axis_error"] == 0
            and status["motor_error"] == 0
            and status["encoder_error"] == 0
        )
        condition = "READY" if ready else "NOT READY"
        torque_constant = status["torque_constant"]
        measured_torque = status["iq_measured"] * torque_constant
        setpoint_torque = status["iq_setpoint"] * torque_constant
        return (
            f"Motor {self.current_motor}: {condition} | state={status['state']} | "
            f"vbus={status['vbus_voltage']:.2f} V | "
            f"axis=0x{status['axis_error']:X} | motor=0x{status['motor_error']:X} | "
            f"encoder=0x{status['encoder_error']:X} | calibrated={status['calibrated']} | "
            f"encoder_ready={status['encoder_ready']} | "
            f"control_mode={status['control_mode']} | input_mode={status['input_mode']} | "
            f"velocity_limit={status['velocity_limit']:.1f} turns/s | "
            f"input_velocity={status['input_velocity']:.3f} turns/s | "
            f"velocity={status['velocity']:.3f} turns/s | "
            f"current_limit={status['current_limit']:.1f} A | "
            f"torque_limit={status['torque_limit']:.2f} Nm | "
            f"torque_constant={status['torque_constant']:.4f} Nm/A | "
            f"velocity_ramp_rate={status['velocity_ramp_rate']:.1f} turns/s² | "
            f"Iq_setpoint={status['iq_setpoint']:.2f} A "
            f"(estimated torque={setpoint_torque:.2f} Nm) | "
            f"Iq_measured={status['iq_measured']:.2f} A "
            f"(estimated torque={measured_torque:.2f} Nm) | "
            f"position={status['position']:.3f} turns"
        )

    def reset_velocity_rate_peak(self):
        self._velocity_samples.clear()
        self._peak_velocity_rate = 0.0
        self._peak_current = 0.0
        self.velocity_rate_label.setText(
            "Actual speed change rate: -- rev/s² | peak: 0.00"
        )
        self.current_feedback_label.setText("Motor current: -- A | peak: 0.00 A")

    def _record_velocity_sample(self, timestamp, velocity):
        self._velocity_samples.append((timestamp, velocity))
        if len(self._velocity_samples) < 2:
            return None
        times = [sample[0] for sample in self._velocity_samples]
        velocities = [sample[1] for sample in self._velocity_samples]
        mean_time = sum(times) / len(times)
        mean_velocity = sum(velocities) / len(velocities)
        denominator = sum((sample_time - mean_time) ** 2 for sample_time in times)
        if denominator <= 0:
            return None
        rate = sum(
            (sample_time - mean_time) * (sample_velocity - mean_velocity)
            for sample_time, sample_velocity in self._velocity_samples
        ) / denominator
        self._peak_velocity_rate = max(self._peak_velocity_rate, abs(rate))
        return rate

    def update_live_velocity_rate(self):
        if self.odrive is None:
            self.reset_velocity_rate_peak()
            return

        try:
            feedback = self.odrive.request_feedback(self.current_motor).split()
            if len(feedback) != 2:
                raise ValueError(f"invalid velocity feedback: {feedback!r}")
            velocity = float(feedback[1])
            current = abs(
                float(
                    self.odrive.read_parameter(
                        f"axis{self.current_motor}.motor.current_control.Iq_measured"
                    )
                )
            )
            self._peak_current = max(self._peak_current, current)
            self.current_feedback_label.setText(
                f"Motor current: {current:.2f} A | peak: {self._peak_current:.2f} A"
            )
            rate = self._record_velocity_sample(time.monotonic(), velocity)
            rate_text = "--" if rate is None else f"{rate:+.2f}"
            self.velocity_rate_label.setText(
                f"Speed change rate: {rate_text} rev/s² | "
                f"peak: {self._peak_velocity_rate:.2f} | speed: {velocity:.3f} rev/s"
            )
        except Exception:
            self._velocity_samples.clear()
            self.velocity_rate_label.setText("Speed change rate: unavailable")

    def read_motor_status(self):
        if self.odrive is None:
            self.feedback_label.setText("ODrive not connected")
            return

        try:
            status = self._read_status()
        except Exception as exc:
            error_message = self._operation_error("read motor status", exc)
            self.feedback_label.setText(error_message)
            QMessageBox.critical(self, "Status read failed", error_message)
        else:
            self.feedback_label.setText(self._format_status(status))

    def clear_odrive_errors(self):
        if self.odrive is None:
            self.feedback_label.setText("ODrive not connected")
            return

        axis = self.current_motor
        try:
            self._stop_motor(axis)

            self.odrive.system_command("sc")
            time.sleep(0.2)
            status = self._read_status()
            active_errors = {
                name: status[name]
                for name in ("axis_error", "motor_error", "encoder_error")
                if status[name]
            }
            if active_errors:
                raise RuntimeError(f"errors remain after clear: {active_errors}")
            if not status["calibrated"]:
                raise RuntimeError("motor is not calibrated")
            if not status["encoder_ready"]:
                raise RuntimeError("encoder is not ready")

            self._set_control_mode(2)
            self.odrive.motor_velocity(axis, 0, 0)
            self.odrive.write_parameter(f"axis{axis}.requested_state", 8)
            time.sleep(0.3)
            status = self._read_status()
            if status["state"] != 8:
                raise RuntimeError(
                    f"motor did not enter closed loop; state={status['state']}"
                )
        except Exception as exc:
            error_message = self._operation_error(f"restore Motor {axis}", exc)
            self.feedback_label.setText(error_message)
            QMessageBox.critical(self, "Closed-loop recovery failed", error_message)
        else:
            self.feedback_label.setText(self._format_status(status))

    def disconnect_odrive(self):
        odrive = self.odrive
        if odrive is None:
            self.feedback_label.setText("ODrive not connected")
            return True

        try:
            closed = odrive.close()
            if closed is False:
                raise OSError("serial port is still open after close")
        except Exception as exc:
            self.feedback_label.setText(
                f"Disconnect failed; serial port may still be in use: {exc}"
            )
            return False

        self.odrive = None
        self.reset_velocity_rate_peak()
        self.configured_control_mode = None
        self.torque_constant = None
        self.torque_limit = 0.0
        self._stored_limits = None
        self.stored_limit_label.setText("ODrive limits: not read")
        self.torque_limit_input.setValue(0.0)
        self._set_connected_state(False)
        self._update_limit_info()
        self.feedback_label.setText("Disconnected; serial port released")
        return True

    def closeEvent(self, event):
        if self.disconnect_odrive():
            event.accept()
        else:
            event.ignore()


def acquire_instance_lock():
    lock_path = os.path.join(tempfile.gettempdir(), "equicycle_motor_control_ui.lock")
    lock = QLockFile(lock_path)
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        return None
    return lock


def main():
    app = QApplication(sys.argv)
    instance_lock = acquire_instance_lock()
    if instance_lock is None:
        QMessageBox.warning(
            None,
            "Motor Control already running",
            "Another Motor Control window is already running. "
            "Close it before starting a new one.",
        )
        return 1
    motor_control_ui = MotorControlUI()
    motor_control_ui.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
