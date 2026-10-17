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
        self.previous_flag_go = balance_control.flag_go
        self.previous_v_vision = balance_control.v_vision
        with balance_control.control_params_lock:
            self.previous_control_params = balance_control.control_params.copy()
            balance_control.control_params.clear()
        balance_control.flag_go = 1
        balance_control.v_vision = None

    def tearDown(self):
        with balance_control.control_params_lock:
            balance_control.control_params.clear()
            balance_control.control_params.update(self.previous_control_params)
        balance_control.flag_go = self.previous_flag_go
        balance_control.v_vision = self.previous_v_vision

    def test_pitch_maps_to_signed_motor_speed(self):
        self.assertEqual(balance_control.pitch_to_motor_speed(0.0), 0.0)
        self.assertEqual(balance_control.pitch_to_motor_speed(3.5), 3.5)
        self.assertEqual(balance_control.pitch_to_motor_speed(-3.5), -3.5)

    def test_pitch_speed_is_limited(self):
        self.assertEqual(
            balance_control.pitch_to_motor_speed(30.0),
            balance_control.MOTOR_SPEED_LIMIT,
        )
        self.assertEqual(
            balance_control.pitch_to_motor_speed(-30.0),
            -balance_control.MOTOR_SPEED_LIMIT,
        )

    def test_non_finite_pitch_is_rejected(self):
        with self.assertRaises(ValueError):
            balance_control.pitch_to_motor_speed(float("nan"))

    def test_control_layer_sends_pitch_direction_to_motor_zero(self):
        for pitch, expected_speed in ((2.0, 2.0), (0.0, 0.0), (-2.0, -2.0)):
            with self.subTest(pitch=pitch):
                balance_control.control_layer(
                    {
                        "device": "ch100",
                        "pitch": pitch,
                        "gyr": [0.0, 0.0, 0.0],
                    },
                    self.odrive,
                )
                balance_control.control_layer(
                    {"device": "odrive", "feedback": "0 0"},
                    self.odrive,
                )

                self.assertEqual(
                    self.odrive.commands[-1],
                    (0, expected_speed, 0),
                )

    def test_excessive_pitch_stops_both_balance_and_drive_motors(self):
        with balance_control.control_params_lock:
            balance_control.control_params["euler_angles"] = {
                "pitch": 8.0,
            }

        balance_control.adjust_motor_speed(self.odrive)

        self.assertIn((0, 0, 0), self.odrive.commands)
        self.assertIn((1, 0, 0), self.odrive.commands)

    def test_invalid_pitch_stops_both_motors_before_raising(self):
        with balance_control.control_params_lock:
            balance_control.control_params["euler_angles"] = {
                "pitch": float("nan"),
            }

        with self.assertRaises(ValueError):
            balance_control.adjust_motor_speed(self.odrive)

        self.assertEqual(self.odrive.commands, [(0, 0, 0), (1, 0, 0)])


if __name__ == "__main__":
    unittest.main()
