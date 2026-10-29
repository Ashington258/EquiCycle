import time
import unittest

from balance_control import balance_control


class RecordingODrive:
    def __init__(self):
        self.commands = []

    def motor_velocity(self, motor, velocity, torque_ff=0.0):
        self.commands.append((motor, velocity, torque_ff))


class PitchControlTests(unittest.TestCase):
    def setUp(self):
        self.odrive = RecordingODrive()
        with balance_control.control_params_lock:
            self.previous_control_params = balance_control.control_params.copy()
            self.previous_mode = balance_control.balance_mode
            balance_control.control_params.clear()
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_DOCKED)

    def tearDown(self):
        with balance_control.control_params_lock:
            balance_control.control_params.clear()
            balance_control.control_params.update(self.previous_control_params)
        balance_control.set_balance_mode(self.previous_mode)

    def test_docked_mode_keeps_motor_stopped_at_measured_dock_angle(self):
        balance_control.process_ch100_data(
            {"pitch": 9.47, "gyr": [0.0, 0.0, 0.0]}
        )
        with balance_control.control_params_lock:
            balance_control.control_params["motor_speed"] = 0.0

        target_speed = balance_control.adjust_motor_speed(self.odrive)

        self.assertEqual(target_speed, 0.0)
        self.assertEqual(self.odrive.commands, [(0, 0, 0)])
        self.assertEqual(
            balance_control.get_balance_status(),
            (balance_control.BALANCE_MODE_DOCKED, 9.47),
        )

    def test_balance_mode_uses_zero_degree_setpoint(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)

        self.assertEqual(
            balance_control.get_balance_status(),
            (balance_control.BALANCE_MODE_ACTIVE, 0.0),
        )

    def test_pitch_status_returns_latest_angle_and_freshness(self):
        balance_control.process_ch100_data(
            {"pitch": -1.25, "gyr": [0.0, 0.0, 0.0]}
        )

        self.assertEqual(balance_control.get_pitch_status(), (-1.25, True))

    def test_pitch_and_gyro_low_pass_filters_smooth_step_input(self):
        balance_control.process_ch100_data(
            {"pitch": 0.0, "gyr": [0.0, 0.0, 0.0]}
        )
        with balance_control.control_params_lock:
            balance_control.control_params["pitch_received_at"] = time.monotonic() - 0.005

        balance_control.process_ch100_data(
            {"pitch": 10.0, "gyr": [100.0, 0.0, 0.0]}
        )
        snapshot = balance_control.get_tuning_snapshot()

        self.assertEqual(snapshot["pitch_raw_deg"], 10.0)
        self.assertGreater(snapshot["pitch_deg"], 0.0)
        self.assertLess(snapshot["pitch_deg"], 10.0)
        self.assertEqual(snapshot["gyro_x_raw_deg_s"], 100.0)
        self.assertGreater(snapshot["gyro_x_filtered_deg_s"], 0.0)
        self.assertLess(snapshot["gyro_x_filtered_deg_s"], 100.0)

    def test_gyro_deadband_suppresses_static_noise(self):
        balance_control.process_ch100_data(
            {"pitch": 0.0, "gyr": [0.05, 0.0, 0.0]}
        )
        snapshot = balance_control.get_tuning_snapshot()

        self.assertEqual(snapshot["gyro_x_filtered_deg_s"], 0.05)
        self.assertEqual(snapshot["gyro_x_deg_s"], 0.0)

    def test_pitch_deadband_suppresses_small_setpoint_error(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        balance_control.process_ch100_data(
            {"pitch": 0.05, "gyr": [0.0, 0.0, 0.0]}
        )
        with balance_control.control_params_lock:
            balance_control.control_params["motor_speed"] = 0.0

        target_speed = balance_control.adjust_motor_speed(self.odrive)

        self.assertEqual(target_speed, 0.0)
        self.assertEqual(self.odrive.commands[-1], (0, 0.0, 0))

    def test_balance_mode_controls_from_pitch_and_velocity_feedback(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        balance_control.control_layer(
            {"device": "ch100", "pitch": 2.0, "gyr": [0.0, 0.0, 0.0]},
            self.odrive,
        )
        target_speed = balance_control.control_layer(
            {"device": "odrive", "feedback": "0 0"}, self.odrive
        )

        self.assertNotEqual(target_speed, 0.0)
        self.assertEqual(self.odrive.commands[-1], (0, target_speed, 0))

    def test_angle_loop_polarity_is_reversed(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        balance_control.process_ch100_data(
            {"pitch": 2.0, "gyr": [0.0, 0.0, 0.0]}
        )
        with balance_control.control_params_lock:
            balance_control.control_params["motor_speed"] = 0.0

        target_speed = balance_control.adjust_motor_speed(self.odrive)
        snapshot = balance_control.get_tuning_snapshot()

        self.assertGreater(snapshot["angle_loop_output"], 0.0)
        self.assertGreater(snapshot["raw_cascade_output_rev_s"], 0.0)
        self.assertLess(target_speed, 0.0)

    def test_control_data_update_does_not_issue_motor_command(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)

        balance_control.update_control_data(
            {"device": "ch100", "pitch": 2.0, "gyr": [0.0, 0.0, 0.0]}
        )
        balance_control.update_control_data(
            {"device": "odrive", "feedback": "0 1.5"}
        )

        self.assertEqual(self.odrive.commands, [])
        self.assertEqual(
            balance_control.get_tuning_snapshot()["motor_speed_rev_s"], 1.5
        )

    def test_first_balance_command_passes_pid_output_at_default_interval(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        balance_control.process_ch100_data(
            {"pitch": 9.47, "gyr": [0.0, 0.0, 0.0]}
        )
        with balance_control.control_params_lock:
            balance_control.control_params["motor_speed"] = 0.0

        target_speed = balance_control.adjust_motor_speed(self.odrive)
        snapshot = balance_control.get_tuning_snapshot()

        self.assertEqual(abs(target_speed), balance_control.MOTOR_SPEED_LIMIT)
        self.assertEqual(balance_control.MOTOR_SPEED_SLEW_RATE, 4000.0)
        self.assertEqual(
            balance_control.CascadePIDclass.velocity_pid.output_limits,
            (-15.0, 15.0),
        )
        self.assertNotEqual(target_speed, 0.0)
        self.assertEqual(snapshot["pitch_deg"], 9.47)
        self.assertEqual(snapshot["gyro_x_deg_s"], 0.0)
        self.assertEqual(snapshot["motor_speed_rev_s"], 0.0)
        self.assertEqual(snapshot["limited_output_rev_s"], target_speed)
        self.assertEqual(
            snapshot["limited_output_rev_s"],
            balance_control.MOTOR_DIRECTION
            * snapshot["raw_cascade_output_rev_s"],
        )

    def test_motor_command_reverses_raw_angular_velocity_loop_output(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        balance_control.process_ch100_data(
            {"pitch": 0.0, "gyr": [1.0, 0.0, 0.0]}
        )
        with balance_control.control_params_lock:
            balance_control.control_params["motor_speed"] = 0.0

        target_speed = balance_control.adjust_motor_speed(self.odrive)
        snapshot = balance_control.get_tuning_snapshot()

        self.assertLess(snapshot["raw_cascade_output_rev_s"], 0.0)
        self.assertGreater(target_speed, 0.0)
        self.assertEqual(
            target_speed,
            balance_control.MOTOR_DIRECTION
            * snapshot["raw_cascade_output_rev_s"],
        )

    def test_balance_command_saturates_symmetrically_at_speed_limit(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        for gyro, expected in (
            (-100.0, -balance_control.MOTOR_SPEED_LIMIT),
            (100.0, balance_control.MOTOR_SPEED_LIMIT),
        ):
            with self.subTest(gyro=gyro):
                with balance_control.control_params_lock:
                    balance_control.control_params.pop("gyro", None)
                    balance_control.control_params.pop("pitch_received_at", None)
                balance_control.process_ch100_data(
                    {"pitch": 11.0, "gyr": [gyro, 0.0, 0.0]}
                )
                with balance_control.control_params_lock:
                    balance_control.control_params["motor_speed"] = 0.0
                    balance_control.control_params["last_control_at"] = None
                target_speed = balance_control.adjust_motor_speed(self.odrive)
                self.assertEqual(target_speed, expected)
                self.assertEqual(self.odrive.commands[-1], (0, expected, 0))

    def test_outer_loop_velocity_target_saturates_symmetrically(self):
        controller = balance_control.CascadedPIDController()
        controller.velocity_pid.kp = 100.0

        positive, _, _ = controller.update(0.0, 0.0, -10.0, 0.02, 0.0)
        controller.reset()
        negative, _, _ = controller.update(0.0, 0.0, 10.0, 0.02, 0.0)

        self.assertEqual(positive, balance_control.MOTOR_SPEED_LIMIT)
        self.assertEqual(negative, -balance_control.MOTOR_SPEED_LIMIT)

    def test_mode_switch_resets_pid_state(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        balance_control.CascadePIDclass.velocity_pid.integral = 123.0
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_DOCKED)
        self.assertEqual(balance_control.CascadePIDclass.velocity_pid.integral, 0.0)

    def test_pid_parameters_report_all_cascade_loops(self):
        parameters = balance_control.get_pid_parameters()

        self.assertEqual(set(parameters), {"velocity", "angle", "gyro"})
        self.assertEqual(parameters["velocity"]["kp"], 0.0)
        self.assertEqual(parameters["velocity"]["ki"], 0.0)
        self.assertEqual(parameters["velocity"]["kd"], 0.0)
        self.assertEqual(parameters["angle"]["kp"], 2.45)
        self.assertEqual(parameters["angle"]["ki"], 0.0)
        self.assertEqual(parameters["angle"]["kd"], 1.5)
        self.assertEqual(parameters["gyro"]["kp"], 2.59)
        self.assertEqual(parameters["gyro"]["ki"], 0.0)
        self.assertEqual(parameters["gyro"]["kd"], 0.12)

    def test_stale_pitch_stops_motor_zero(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        with balance_control.control_params_lock:
            balance_control.control_params["euler_angles"] = {"pitch": 5.0}
            balance_control.control_params["pitch_received_at"] = (
                time.monotonic() - balance_control.PITCH_STALE_TIMEOUT - 0.1
            )

        target_speed = balance_control.adjust_motor_speed(self.odrive)

        self.assertEqual(target_speed, 0.0)
        self.assertEqual(self.odrive.commands, [(0, 0, 0)])

    def test_excessive_pitch_stops_motor_zero(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        with balance_control.control_params_lock:
            balance_control.control_params["euler_angles"] = {"pitch": 13.0}
            balance_control.control_params["pitch_received_at"] = time.monotonic()

        balance_control.adjust_motor_speed(self.odrive)

        self.assertEqual(self.odrive.commands, [(0, 0, 0)])

    def test_invalid_pitch_stops_motor_zero_before_raising(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        with balance_control.control_params_lock:
            balance_control.control_params["euler_angles"] = {"pitch": float("nan")}
            balance_control.control_params["pitch_received_at"] = time.monotonic()

        with self.assertRaises(ValueError):
            balance_control.adjust_motor_speed(self.odrive)

        self.assertEqual(self.odrive.commands, [(0, 0, 0)])


if __name__ == "__main__":
    unittest.main()
