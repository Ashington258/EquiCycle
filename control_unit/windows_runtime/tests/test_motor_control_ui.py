import os
import sys
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

from motor_control_ui import MotorControlUI, acquire_instance_lock


class MotorControlUILogicTests(unittest.TestCase):
    def test_velocity_control_uses_ramp_input_mode(self):
        window = MotorControlUI.__new__(MotorControlUI)
        window.current_motor = 0
        window.odrive = Mock()
        window.configured_control_mode = None

        window._set_control_mode(2)

        self.assertEqual(
            window.odrive.write_parameter.call_args_list,
            [
                unittest.mock.call("axis0.controller.config.vel_limit", 35.0),
                unittest.mock.call("axis0.controller.config.input_mode", 2),
                unittest.mock.call("axis0.controller.config.control_mode", 2),
            ],
        )

    def test_reentering_velocity_mode_repairs_external_mode_change(self):
        window = MotorControlUI.__new__(MotorControlUI)
        window.current_motor = 0
        window.odrive = Mock()
        window.configured_control_mode = 2

        window._set_control_mode(2)

        self.assertIn(
            unittest.mock.call("axis0.controller.config.control_mode", 2),
            window.odrive.write_parameter.call_args_list,
        )

    def test_limited_rate_test_uses_selected_limits_and_ramp_mode(self):
        window = MotorControlUI.__new__(MotorControlUI)
        window.current_motor = 0
        window.odrive = Mock()
        window.odrive.read_parameter.side_effect = [
            "30", "1", "60", "15", "2", "2", "20", "1.0", "60", "60", "2", "2"
        ]
        window.current_limit_input = Mock()
        window.current_limit_input.value.return_value = 20.0
        window.torque_limit_input = Mock()
        window.torque_limit_input.value.return_value = 1.0
        window.velocity_ramp_rate_input = Mock()
        window.velocity_ramp_rate_input.value.return_value = 60.0
        window.most_speed_test_slider = Mock()
        window.most_speed_test_slider.value.return_value = 150
        window.telemetry_timer = Mock()
        window.feedback_label = Mock()
        window.torque_constant = 0.1
        window.configured_control_mode = None
        window._velocity_samples = []
        window._peak_velocity_rate = 0.0
        window.reset_velocity_rate_peak = Mock()
        window._read_int = Mock(side_effect=[2, 2, 8, 0, 0, 0])
        window._measure_velocity_transition = Mock(
            return_value={
                "start_velocity": -15.0,
                "target_velocity": 15.0,
                "peak_rate": 80.0,
                "transition_time": 0.875,
                "peak_current": 39.0,
                "peak_setpoint_current": 20.0,
                "final_velocity": 20.0,
                "final_input_velocity": 20.0,
                "samples": 10,
            }
        )

        with patch("motor_control_ui.QMessageBox.question", return_value=QMessageBox.Yes), patch(
            "motor_control_ui.time.sleep"
        ):
            window.test_max_velocity_rate()

        self.assertIn(
            unittest.mock.call("axis0.motor.config.current_lim", 20.0),
            window.odrive.write_parameter.call_args_list,
        )
        self.assertIn(
            unittest.mock.call("axis0.controller.config.vel_ramp_rate", 60.0),
            window.odrive.write_parameter.call_args_list,
        )
        self.assertIn(
            unittest.mock.call("axis0.controller.config.input_mode", 2),
            window.odrive.write_parameter.call_args_list,
        )
        self.assertIn(
            unittest.mock.call("axis0.controller.config.vel_limit", 15.0),
            window.odrive.write_parameter.call_args_list,
        )
        self.assertIn(
            unittest.mock.call("axis0.motor.config.torque_lim", 1.0),
            window.odrive.write_parameter.call_args_list,
        )
        self.assertEqual(window._measure_velocity_transition.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in window._measure_velocity_transition.call_args_list],
            [-15.0, 15.0, -15.0],
        )
        for call in window._measure_velocity_transition.call_args_list:
            self.assertTrue(call.kwargs["update_peak_display"])
        self.assertIn(
            unittest.mock.call("axis0.controller.config.vel_limit", 15.0),
            window.odrive.write_parameter.call_args_list,
        )

    @patch("motor_control_ui.QMessageBox.critical")
    def test_limited_rate_test_stops_when_initial_target_is_not_reached(self, critical):
        window = MotorControlUI.__new__(MotorControlUI)
        window.current_motor = 0
        window.odrive = Mock()
        window.odrive.read_parameter.side_effect = [
            "30", "1", "60", "15", "2", "2", "20", "1.0", "60", "60", "2", "2"
        ]
        window.current_limit_input = Mock()
        window.current_limit_input.value.return_value = 20.0
        window.torque_limit_input = Mock()
        window.torque_limit_input.value.return_value = 1.0
        window.velocity_ramp_rate_input = Mock()
        window.velocity_ramp_rate_input.value.return_value = 60.0
        window.most_speed_test_slider = Mock()
        window.most_speed_test_slider.value.return_value = 150
        window.telemetry_timer = Mock()
        window.feedback_label = Mock()
        window.torque_constant = 0.1
        window.configured_control_mode = None
        window._velocity_samples = []
        window._peak_velocity_rate = 0.0
        window.reset_velocity_rate_peak = Mock()
        window._read_int = Mock(side_effect=[2, 2, 8, 8, 8, 8, 8, 8, 0, 0, 0])
        window._measure_velocity_transition = Mock(
            return_value={
                "start_velocity": 0.0,
                "target_velocity": -20.0,
                "peak_rate": 0.79,
                "transition_time": None,
                "peak_current": 0.5,
                "peak_setpoint_current": 0.6,
                "final_velocity": -1.6,
                "final_input_velocity": -20.0,
                "samples": 20,
            }
        )

        with patch(
            "motor_control_ui.QMessageBox.question", return_value=QMessageBox.Yes
        ), patch("motor_control_ui.time.sleep"):
            window.test_max_velocity_rate()

        window._measure_velocity_transition.assert_called_once()
        failure_message = window.feedback_label.setText.call_args_list[-1].args[0]
        self.assertIn("timed out before reaching target", failure_message)
        self.assertIn("ODrive input_vel=-20.00 rev/s", failure_message)
        self.assertIn("actual=-1.60 rev/s", failure_message)
        self.assertIn("peak Iq_setpoint=0.60 A", failure_message)
        self.assertIn("peak Iq_measured=0.50 A", failure_message)
        critical.assert_called_once()

    def test_most_speed_label_shows_selected_sequence(self):
        window = MotorControlUI.__new__(MotorControlUI)
        window.most_speed_test_label = Mock()

        window.update_most_speed_test_label(125)

        window.most_speed_test_label.setText.assert_called_once_with(
            "most_speed+test: 12.5 rev/s (-12.5 → +12.5 → -12.5)"
        )

    def test_velocity_rate_uses_window_slope_and_tracks_peak(self):
        from collections import deque

        window = MotorControlUI.__new__(MotorControlUI)
        window._velocity_samples = deque(maxlen=5)
        window._peak_velocity_rate = 0.0
        rates = [
            window._record_velocity_sample(1.0, 1.0),
            window._record_velocity_sample(1.1, 1.4),
            window._record_velocity_sample(1.2, 1.8),
        ]

        self.assertIsNone(rates[0])
        self.assertAlmostEqual(rates[-1], 4.0)
        self.assertAlmostEqual(window._peak_velocity_rate, 4.0)


class MotorControlUISingleInstanceTests(unittest.TestCase):
    def test_second_instance_lock_is_rejected(self):
        first_lock = acquire_instance_lock()
        self.assertIsNotNone(first_lock)
        try:
            self.assertIsNone(acquire_instance_lock())
        finally:
            first_lock.unlock()



class MotorControlUIConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.window = MotorControlUI()
        self.window.port_input.setText("COM27")

    def tearDown(self):
        self.window.close()

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_disconnect_releases_port_and_allows_reconnect(self, protocol_class):
        first_drive = Mock()
        first_drive.read_parameter.return_value = "24.0"
        second_drive = Mock()
        second_drive.read_parameter.return_value = "24.0"
        protocol_class.side_effect = [first_drive, second_drive]

        self.window.connect_odrive()

        self.assertIs(self.window.odrive, first_drive)
        self.assertFalse(self.window.connect_button.isEnabled())
        self.assertTrue(self.window.disconnect_button.isEnabled())
        self.assertTrue(self.window.read_status_button.isEnabled())
        self.assertTrue(self.window.clear_errors_button.isEnabled())
        self.assertFalse(self.window.port_input.isEnabled())

        self.window.disconnect_odrive()

        first_drive.close.assert_called_once_with()
        self.assertIsNone(self.window.odrive)
        self.assertTrue(self.window.connect_button.isEnabled())
        self.assertFalse(self.window.disconnect_button.isEnabled())
        self.assertFalse(self.window.read_status_button.isEnabled())
        self.assertFalse(self.window.clear_errors_button.isEnabled())
        self.assertTrue(self.window.port_input.isEnabled())

        self.window.connect_odrive()

        self.assertIs(self.window.odrive, second_drive)
        self.assertEqual(protocol_class.call_count, 2)

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_close_error_keeps_connection_available_for_retry(self, protocol_class):
        failed_drive = Mock()
        failed_drive.close.side_effect = OSError("close failed")
        failed_drive.read_parameter.return_value = "24.0"
        replacement_drive = Mock()
        replacement_drive.read_parameter.return_value = "24.0"
        protocol_class.side_effect = [failed_drive, replacement_drive]

        self.window.connect_odrive()
        self.window.disconnect_odrive()

        self.assertIn("close failed", self.window.feedback_label.text())
        self.assertIs(self.window.odrive, failed_drive)
        self.assertFalse(self.window.connect_button.isEnabled())

        failed_drive.close.side_effect = None
        failed_drive.close.return_value = True
        self.window.disconnect_odrive()
        self.window.connect_odrive()

        self.assertIs(self.window.odrive, replacement_drive)

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_disconnect_keeps_connected_state_when_port_remains_open(
        self, protocol_class
    ):
        drive = protocol_class.return_value
        drive.read_parameter.return_value = "24.0"
        drive.close.return_value = False

        self.window.connect_odrive()
        disconnected = self.window.disconnect_odrive()

        self.assertFalse(disconnected)
        self.assertIs(self.window.odrive, drive)
        self.assertFalse(self.window.connect_button.isEnabled())
        self.assertTrue(self.window.disconnect_button.isEnabled())
        self.assertIn("may still be in use", self.window.feedback_label.text())

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_repeated_connect_does_not_open_port_twice(self, protocol_class):
        protocol_class.return_value = Mock()
        protocol_class.return_value.read_parameter.return_value = "24.0"

        self.window.connect_odrive()
        self.window.connect_odrive()

        protocol_class.assert_called_once_with("COM27")

    @patch("motor_control_ui.QMessageBox.critical")
    @patch("motor_control_ui.ODriveAsciiProtocol", side_effect=OSError("access denied"))
    def test_connection_failure_displays_detailed_error(self, _protocol, critical):
        self.window.connect_odrive()

        message = self.window.feedback_label.text()
        self.assertIn("COM27", message)
        self.assertIn("OSError", message)
        self.assertIn("access denied", message)
        critical.assert_called_once()

    @patch("motor_control_ui.time.sleep")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_clear_errors_restores_velocity_closed_loop(self, protocol_class, _sleep):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = [
            "24.0", "10.0", "0.1", "1.0", "60.0",
            "1",
            "24.0", "1", "0", "0", "0", "1", "1", "3", "1", "15", "0", "10", "1.0", "0.1", "60", "0", "0",
            "24.0", "8", "0", "0", "0", "1", "1", "2", "2", "15", "0", "10", "1.0", "0.1", "60", "0", "0",
        ]
        drive.request_feedback.side_effect = ["267.837 0", "267.837 0"]
        self.window.connect_odrive()

        self.window.clear_odrive_errors()

        drive.system_command.assert_called_once_with("sc")
        self.assertEqual(
            drive.write_parameter.call_args_list,
            [
                unittest.mock.call("axis0.requested_state", 1),
                unittest.mock.call("axis0.controller.config.vel_limit", 35.0),
                unittest.mock.call("axis0.controller.config.input_mode", 2),
                unittest.mock.call("axis0.controller.config.control_mode", 2),
                unittest.mock.call("axis0.requested_state", 8),
            ],
        )
        self.assertEqual(drive.motor_velocity.call_count, 2)
        self.assertIn("READY", self.window.feedback_label.text())

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_read_status_displays_axis_state_and_errors(self, protocol_class):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = [
            "24.0", "10.0", "0.1", "1.0", "60.0",
            "24.1", "1", "256", "0", "4", "1", "0", "3", "1",
            "15", "12.5", "10", "1.0", "0.1", "60.0", "2.5", "3.0"
        ]
        drive.request_feedback.return_value = "267.837 0"
        self.window.connect_odrive()

        self.window.read_motor_status()

        message = self.window.feedback_label.text()
        self.assertIn("NOT READY", message)
        self.assertIn("axis=0x100", message)
        self.assertIn("encoder=0x4", message)
        self.assertIn("vbus=24.10 V", message)
        self.assertIn("velocity_limit=15.0 turns/s", message)
        self.assertIn("input_velocity=12.500 turns/s", message)
        self.assertIn("torque_limit=1.00 Nm", message)
        self.assertIn("velocity_ramp_rate=60.0 turns/s²", message)
        self.assertIn(
            "Iq_measured=2.50 A (estimated torque=0.25 Nm)", message
        )

    @patch("motor_control_ui.QMessageBox.critical")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_failed_handshake_closes_serial_connection(self, protocol_class, _critical):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = TimeoutError("no response")

        self.window.connect_odrive()

        drive.close.assert_called_once_with()
        self.assertIsNone(self.window.odrive)
        self.assertIn("no response", self.window.feedback_label.text())

    def test_default_test_limits_match_runtime_configuration(self):
        self.assertEqual(self.window.current_limit_input.value(), 30.0)
        self.assertEqual(self.window.torque_limit_input.value(), 1.0)
        self.assertFalse(self.window.torque_limit_input.isReadOnly())
        self.assertEqual(self.window.velocity_ramp_rate_input.value(), 60.0)
        self.assertEqual(self.window.most_speed_test_slider.maximum(), 350)
        self.assertEqual(self.window.velocity_ramp_rate_input.singleStep(), 10.0)

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_current_limit_does_not_change_independent_torque_limit(self, protocol_class):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = ["24.0", "10.0", "0.1", "1.0", "60.0"]
        self.window.connect_odrive()

        self.window.current_limit_input.setValue(20.0)

        self.assertEqual(self.window.torque_limit_input.value(), 1.0)
        self.assertEqual(self.window.torque_limit, 1.0)
        self.assertIn("torque limit: 1.00 Nm", self.window.limit_info_label.text())

    def test_velocity_rate_uses_window_slope_and_tracks_peak(self):
        rates = [
            self.window._record_velocity_sample(1.0, 1.0),
            self.window._record_velocity_sample(1.1, 1.4),
            self.window._record_velocity_sample(1.2, 1.8),
        ]

        self.assertIsNone(rates[0])
        self.assertAlmostEqual(rates[-1], 4.0)
        self.assertAlmostEqual(self.window._peak_velocity_rate, 4.0)
        self.window.reset_velocity_rate_peak()
        self.assertEqual(self.window._peak_velocity_rate, 0.0)
        self.assertEqual(len(self.window._velocity_samples), 0)

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_connection_displays_current_limit_above_old_ui_ceiling(self, protocol_class):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = ["24.0", "120.0", "0.1", "12.0", "60.0"]

        self.window.connect_odrive()

        self.assertEqual(self.window.current_limit_input.value(), 100.0)
        self.assertIn("current=120.0 A", self.window.stored_limit_label.text())
        self.assertIn("torque=12.00 Nm", self.window.stored_limit_label.text())
        drive.write_parameter.assert_not_called()

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_position_command_does_not_use_unapplied_higher_torque(self, protocol_class):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = ["24.0", "10.0", "0.1", "0.4", "60.0"]
        self.window.connect_odrive()
        self.window.torque_limit_input.setValue(1.0)

        self.window.slider_position_changed(100)

        drive.motor_position.assert_called_once_with(
            0, 1.0, velocity_lim=35.0, torque_lim=0.4
        )
        self.assertIn("ramp mode not selected", self.window.stored_limit_label.text())

    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_sliders_switch_control_mode_before_commands(self, protocol_class):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = ["24.0", "10.0", "0.1", "1.0", "60.0"]
        self.window.connect_odrive()

        self.assertEqual(self.window.velocity_slider.minimum(), -350)
        self.assertEqual(self.window.velocity_slider.maximum(), 350)
        self.assertEqual(self.window.current_limit_input.maximum(), 150.0)
        self.assertEqual(self.window.torque_limit_input.maximum(), 40.0)
        self.assertEqual(
            self.window.velocity_ramp_rate_input.maximum(), 400.0
        )
        self.window.slider_position_changed(150)
        self.window.slider_velocity_changed(20)

        self.assertEqual(
            drive.write_parameter.call_args_list,
            [
                unittest.mock.call("axis0.controller.config.vel_limit", 35.0),
                unittest.mock.call("axis0.controller.config.input_mode", 1),
                unittest.mock.call("axis0.controller.config.control_mode", 3),
                unittest.mock.call("axis0.controller.config.vel_limit", 35.0),
                unittest.mock.call("axis0.controller.config.input_mode", 2),
                unittest.mock.call("axis0.controller.config.control_mode", 2),
            ],
        )
        drive.motor_position.assert_called_once_with(
            0, 1.5, velocity_lim=35.0, torque_lim=1.0
        )
        drive.motor_velocity.assert_called_once_with(0, 2.0)

    @patch("motor_control_ui.QMessageBox.critical")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_slider_command_failure_is_displayed(self, protocol_class, critical):
        drive = protocol_class.return_value
        drive.read_parameter.return_value = "24.0"
        drive.write_parameter.side_effect = OSError("serial write failed")
        self.window.connect_odrive()

        self.window.slider_velocity_changed(10)

        self.assertIn("serial write failed", self.window.feedback_label.text())
        critical.assert_called_once()

    @patch("motor_control_ui.time.sleep")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_apply_limits_uses_idle_and_verifies_readback(self, protocol_class, _sleep):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = [
            "24.0", "10.0", "0.1", "1.0", "60.0",
            "1", "20.0", "1.0", "60.0", "35.0", "2", "2", "8"
        ]
        self.window.connect_odrive()
        self.window.current_limit_input.setValue(150.0)
        self.window.torque_limit_input.setValue(40.0)
        self.window.velocity_ramp_rate_input.setValue(60.0)

        self.window.apply_motor_limits()

        self.assertEqual(
            drive.write_parameter.call_args_list,
            [
                unittest.mock.call("axis0.requested_state", 1),
                unittest.mock.call("axis0.motor.config.current_lim", 150.0),
                unittest.mock.call("axis0.motor.config.torque_lim", 40.0),
                unittest.mock.call(
                    "axis0.controller.config.vel_ramp_rate", 60.0
                ),
                unittest.mock.call("axis0.controller.config.vel_limit", 35.0),
                unittest.mock.call("axis0.controller.config.input_mode", 2),
                unittest.mock.call("axis0.controller.config.control_mode", 2),
                unittest.mock.call("axis0.requested_state", 8),
            ],
        )
        self.assertEqual(drive.motor_velocity.call_count, 2)
        self.assertIn("current=20.0 A", self.window.feedback_label.text())
        self.assertIn("velocity limit=35.0 rev/s", self.window.feedback_label.text())
        self.assertIn(
            "velocity change limit=60.0 rev/s²",
            self.window.feedback_label.text(),
        )
        self.assertIn(
            unittest.mock.call("axis0.motor.config.torque_lim", 1.0),
            drive.write_parameter.call_args_list,
        )

    @patch("motor_control_ui.QMessageBox.critical")
    @patch("motor_control_ui.time.sleep")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_velocity_change_limit_readback_mismatch_stops_apply(
        self, protocol_class, _sleep, critical
    ):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = [
            "24.0", "10.0", "0.1", "1.0", "60.0",
            "1", "20.0", "1.0", "30.0"
        ]
        self.window.connect_odrive()
        self.window.current_limit_input.setValue(20.0)
        self.window.velocity_ramp_rate_input.setValue(60.0)

        self.window.apply_motor_limits()

        self.assertIn(
            "velocity change limit readback mismatch",
            self.window.feedback_label.text(),
        )
        self.assertNotIn(
            unittest.mock.call("axis0.requested_state", 8),
            drive.write_parameter.call_args_list,
        )
        critical.assert_called_once()

    @patch("motor_control_ui.QMessageBox.critical")
    @patch("motor_control_ui.time.sleep")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_velocity_ramp_mode_readback_mismatch_stops_apply(
        self, protocol_class, _sleep, critical
    ):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = [
            "24.0", "10.0", "0.1", "1.0", "60.0",
            "1", "20.0", "1.0", "60.0", "35.0", "1"
        ]
        self.window.connect_odrive()
        self.window.current_limit_input.setValue(20.0)
        self.window.torque_limit_input.setValue(1.0)
        self.window.velocity_ramp_rate_input.setValue(60.0)

        self.window.apply_motor_limits()

        self.assertIn(
            "velocity ramp mode readback mismatch",
            self.window.feedback_label.text(),
        )
        self.assertNotIn(
            unittest.mock.call("axis0.requested_state", 8),
            drive.write_parameter.call_args_list,
        )
        critical.assert_called_once()

    @patch("motor_control_ui.QMessageBox.critical")
    @patch("motor_control_ui.time.sleep")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_velocity_limit_readback_mismatch_stops_apply(
        self, protocol_class, _sleep, critical
    ):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = [
            "24.0", "10.0", "0.1", "1.0", "60.0",
            "1", "20.0", "1.0", "60.0", "2.0"
        ]
        self.window.connect_odrive()
        self.window.current_limit_input.setValue(20.0)

        self.window.apply_motor_limits()

        self.assertIn("velocity limit readback mismatch", self.window.feedback_label.text())
        self.assertNotIn(
            unittest.mock.call("axis0.requested_state", 8),
            drive.write_parameter.call_args_list,
        )
        critical.assert_called_once()

    @patch("motor_control_ui.time.sleep")
    @patch("motor_control_ui.ODriveAsciiProtocol")
    def test_apply_limits_uses_explicit_torque_limit(
        self, protocol_class, _sleep
    ):
        drive = protocol_class.return_value
        drive.read_parameter.side_effect = [
            "24.0", "10.0", "0.1", "1.0", "60.0",
            "1", "10.0", "1.0", "60.0", "35.0", "2", "2", "8"
        ]
        self.window.connect_odrive()
        self.window.current_limit_input.setValue(10.0)
        self.window.velocity_ramp_rate_input.setValue(60.0)

        self.window.apply_motor_limits()
        self.window.slider_position_changed(100)

        drive.motor_position.assert_called_once_with(
            0, 1.0, velocity_lim=35.0, torque_lim=1.0
        )
        self.assertIn(
            unittest.mock.call("axis0.motor.config.torque_lim", 1.0),
            drive.write_parameter.call_args_list,
        )


if __name__ == "__main__":
    unittest.main()



