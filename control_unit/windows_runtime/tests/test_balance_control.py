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
        self.assertNotEqual(target_speed, 0.0)
        self.assertEqual(snapshot["pitch_deg"], 9.47)
        self.assertEqual(snapshot["gyro_x_deg_s"], 0.0)
        self.assertEqual(snapshot["motor_speed_rev_s"], 0.0)
        self.assertEqual(snapshot["limited_output_rev_s"], target_speed)
        self.assertEqual(
            snapshot["raw_cascade_output_rev_s"],
            snapshot["limited_output_rev_s"],
        )

    def test_balance_command_never_exceeds_speed_limit(self):
        balance_control.set_balance_mode(balance_control.BALANCE_MODE_ACTIVE)
        balance_control.process_ch100_data(
            {"pitch": 11.0, "gyr": [-100.0, 0.0, 0.0]}
        )
        with balance_control.control_params_lock:
            balance_control.control_params["motor_speed"] = 20.0

        target_speed = balance_control.adjust_motor_speed(self.odrive)

        self.assertEqual(target_speed, balance_control.MOTOR_SPEED_LIMIT)
        self.assertEqual(self.odrive.commands[-1], (0, balance_control.MOTOR_SPEED_LIMIT, 0))

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
