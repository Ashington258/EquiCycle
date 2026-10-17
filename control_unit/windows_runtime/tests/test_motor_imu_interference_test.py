import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import motor_imu_interference_test as interference_test


class FakeDrive:
    def __init__(self, parameters=None, feedback="1.5 2.5"):
        self.parameters = parameters or {
            "axis0.current_state": "8",
            "axis0.error": "0",
            "axis0.motor.error": "0",
            "axis0.encoder.error": "0",
            "axis0.encoder.is_ready": "1",
            "axis0.controller.config.control_mode": "2",
            "axis0.controller.config.input_mode": "2",
        }
        self.feedback = feedback
        self.commands = []

    def read_parameter(self, name):
        return self.parameters[name]

    def request_feedback(self, motor):
        return self.feedback

    def motor_velocity(self, motor, velocity, torque_ff=0.0):
        self.commands.append((motor, velocity, torque_ff))


class FakeCH100Device:
    def __init__(self, frames):
        self.frames = frames
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def read_and_parse(self):
        if self.frames:
            return [self.frames.pop(0)]
        return []


class MotorImuInterferenceTests(unittest.TestCase):
    def test_stage_definition_matches_requested_sequence(self):
        self.assertEqual(
            interference_test.STAGES,
            (
                (1, 0.0, 10.0),
                (2, 5.0, 10.0),
                (3, 10.0, 10.0),
                (4, 15.0, 10.0),
                (5, 20.0, 10.0),
                (6, 25.0, 10.0),
                (7, 0.0, 20.0),
            ),
        )

    def test_axis_check_accepts_velocity_control(self):
        values = interference_test.check_axis_ready(FakeDrive())
        self.assertEqual(values["control_mode"], 2)
        self.assertEqual(values["input_mode"], 2)

    def test_axis_check_rejects_position_control(self):
        drive = FakeDrive()
        drive.parameters["axis0.controller.config.control_mode"] = "3"
        with self.assertRaisesRegex(RuntimeError, "速度控制模式"):
            interference_test.check_axis_ready(drive)

    def test_parse_feedback_requires_two_finite_numbers(self):
        self.assertEqual(
            interference_test.parse_feedback("1.25 -2.5"), (1.25, -2.5)
        )
        with self.assertRaises(RuntimeError):
            interference_test.parse_feedback("1.25")
        with self.assertRaises(RuntimeError):
            interference_test.parse_feedback("nan 0")

    def test_imu_worker_writes_angles_and_motor_state(self):
        state = interference_test.TestState()
        with patch(
            "motor_imu_interference_test.time.monotonic", return_value=100.0
        ):
            state.start_stage(3, 10.0)
            state.set_actual_speed(9.75)
        stop_event = threading.Event()
        first_frame_event = threading.Event()
        errors = []
        device = FakeCH100Device(
            [
                {
                    "roll": 1.0,
                    "pitch": 2.0,
                    "yaw": 3.0,
                    "gyr": (4.0, 5.0, 6.0),
                    "acc": (7.0, 8.0, 9.0),
                    "system_time_ms": 42,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "result.csv"

            def stop_after_empty_read():
                if device.frames:
                    return [device.frames.pop(0)]
                stop_event.set()
                return []

            device.read_and_parse = stop_after_empty_read
            with patch(
                "motor_imu_interference_test.time.monotonic", return_value=101.5
            ):
                interference_test.imu_logging_worker(
                    device,
                    state,
                    log_path,
                    stop_event,
                    first_frame_event,
                    errors,
                )
            csv_text = log_path.read_text(encoding="utf-8-sig")

        self.assertTrue(first_frame_event.is_set())
        self.assertFalse(errors)
        self.assertIn("stage,target_speed_rev_s,actual_speed_rev_s", csv_text)
        self.assertIn("roll_deg,pitch_deg,yaw_deg,gyr_x,gyr_y,gyr_z,acc_x,acc_y,acc_z", csv_text)
        self.assertIn(
            "1.500000,3,10.000,9.750000,1.000000,2.000000,3.000000,"
            "4.000000,5.000000,6.000000,7.000000,8.000000,9.000000,42",
            csv_text,
        )
        self.assertTrue(device.opened)
        self.assertTrue(device.closed)


if __name__ == "__main__":
    unittest.main()
