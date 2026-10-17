import csv
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class FakeODrive:
    def __init__(self, overrides=None):
        self.values = {
            "error": "0",
            "axis0.error": "0",
            "axis0.motor.error": "0",
            "axis0.encoder.error": "0",
            "axis0.motor.is_calibrated": "1",
            "axis0.encoder.is_ready": "1",
            "axis0.current_state": "8",
            "axis0.motor.config.current_lim": "10.0",
        }
        if overrides:
            self.values.update(overrides)
        self.commands = []

    def read_parameter(self, name):
        return self.values[name]

    def request_feedback(self, axis):
        return "0 0"

    def write_parameter(self, name, value):
        self.commands.append(("write", name, value))
        if name == "axis0.requested_state":
            self.values["axis0.current_state"] = str(value)
        else:
            self.values[name] = str(value)

    def system_command(self, command):
        self.commands.append(("system", command))
        if command == "sc":
            self.values["axis0.error"] = "0"
            self.values["axis0.motor.error"] = "0"
            self.values["axis0.encoder.error"] = "0"

    def motor_velocity(self, motor, velocity, torque_ff=0.0):
        self.commands.append(("velocity", motor, velocity, torque_ff))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class MainAxisControlTests(unittest.TestCase):
    def test_feedback_thread_retries_current_limit_read_failure(self):
        from unittest.mock import Mock

        drive = Mock()
        drive.read_parameter.side_effect = [
            TimeoutError("serial timeout"), "10", "1.5", "2", "3", "8", "0", "0", "0"
        ]
        event = Mock()
        event.is_set.side_effect = [False, False, True]
        with patch.object(main, "stop_event", event), patch.object(
            main.logging, "exception"
        ) as log_error, patch.object(main, "data_queue") as queue:
            main.odrive_thread_function(drive)

        log_error.assert_called_once()
        self.assertEqual(event.wait.call_count, 2)
        event.wait.assert_called_with(0.02)
        queue.put.assert_called_once()
        self.assertEqual(queue.put.call_args.args[0]["odrive_current_limit_a"], 10.0)
        self.assertEqual(queue.put.call_args.args[0]["iq_setpoint_a"], 2.0)

    @patch("main.time.sleep", return_value=None)
    def test_prepare_axis_enters_velocity_closed_loop(self, _sleep):
        drive = FakeODrive()

        main.prepare_axis0_closed_loop(drive)

        self.assertEqual(
            drive.commands,
            [
                ("velocity", 0, 0, 0),
                ("write", "axis0.motor.config.current_lim", 90.0),
                ("write", "axis0.controller.config.input_mode", 1),
                ("write", "axis0.controller.config.control_mode", 2),
                ("write", "axis0.requested_state", 8),
            ],
        )

    @patch("main.time.sleep", return_value=None)
    def test_prepare_rejects_current_limit_readback_mismatch(self, _sleep):
        drive = FakeODrive()
        original_write = drive.write_parameter

        def ignore_current_limit(name, value):
            if name == "axis0.motor.config.current_lim":
                drive.commands.append(("write", name, value))
                return
            original_write(name, value)

        drive.write_parameter = ignore_current_limit

        with self.assertRaisesRegex(RuntimeError, "电流限制写入验证失败"):
            main.prepare_axis0_closed_loop(drive)

        self.assertNotIn(("write", "axis0.requested_state", 8), drive.commands)

    def test_prepare_rejects_system_level_fault_even_while_idle(self):
        for state in ("1", "8"):
            with self.subTest(state=state):
                drive = FakeODrive(
                    {
                        "axis0.current_state": state,
                        "axis0.error": "64",
                        "axis0.motor.error": "16777216",
                    }
                )
                with self.assertRaisesRegex(RuntimeError, "16777216"):
                    main.prepare_axis0_closed_loop(drive)
                self.assertEqual(drive.commands, [])

    def test_fault_reports_top_level_error(self):
        drive = FakeODrive({"axis0.error": "64", "error": "256"})

        with self.assertRaisesRegex(RuntimeError, r"ODrive error=256 \(0x100\)"):
            main.prepare_axis0_closed_loop(drive)

        self.assertEqual(drive.commands, [])

    def test_fault_preserved_when_top_level_error_unavailable(self):
        drive = FakeODrive({"axis0.motor.error": "16777216"})
        del drive.values["error"]

        with self.assertRaisesRegex(RuntimeError, "16777216.*顶层 error 读取失败"):
            main.prepare_axis0_closed_loop(drive)

        self.assertEqual(drive.commands, [])

    @patch("main.time.sleep", return_value=None)
    def test_healthy_start_does_not_require_top_level_error(self, _sleep):
        drive = FakeODrive()
        with patch.object(drive, "read_parameter", wraps=drive.read_parameter) as read:
            main.prepare_axis0_closed_loop(drive)

        self.assertNotIn(unittest.mock.call("error"), read.call_args_list)

    def test_prepare_and_stop_twice_without_reboot(self):
        drive = FakeODrive({"axis0.current_state": "1"})
        clock = FakeClock()
        with patch("main.time.monotonic", side_effect=clock.monotonic), patch(
            "main.time.sleep", side_effect=clock.sleep
        ):
            for _ in range(2):
                main.prepare_axis0_closed_loop(drive)
                self.assertEqual(drive.values["axis0.current_state"], "8")
                main.stop_axis0_and_enter_idle(drive)
                self.assertEqual(drive.values["axis0.current_state"], "1")
        states = [command[2] for command in drive.commands
                  if command[:2] == ("write", "axis0.requested_state")]
        self.assertEqual(states, [8, 1, 8, 1])
        self.assertEqual(
            [command for command in drive.commands if command[0] == "system"],
            [("system", "sc"), ("system", "sc")],
        )

    def test_stop_reports_failure_to_enter_idle(self):
        drive = FakeODrive()
        clock = FakeClock()
        drive.write_parameter = lambda name, value: drive.commands.append(
            ("write", name, value)
        )
        with patch("main.time.monotonic", side_effect=clock.monotonic), patch(
            "main.time.sleep", side_effect=clock.sleep
        ):
            with self.assertRaisesRegex(RuntimeError, "未进入 IDLE"):
                main.stop_axis0_and_enter_idle(drive)
        self.assertEqual(drive.commands[-1], ("write", "axis0.requested_state", 1))

    @patch("main.time.sleep", return_value=None)
    def test_prepare_axis_rejects_encoder_error(self, _sleep):
        drive = FakeODrive({"axis0.encoder.error": "2"})

        with self.assertRaisesRegex(RuntimeError, "encoder.error"):
            main.prepare_axis0_closed_loop(drive)

        self.assertEqual(drive.commands, [])

    def test_stop_axis_ramps_measured_speed_before_zero(self):
        drive = FakeODrive()
        clock = FakeClock()
        drive.request_feedback = lambda axis: (
            f"0 {min(0.0, -3.0 + clock.now * 5.0)}"
        )

        with patch("main.time.monotonic", side_effect=clock.monotonic), patch(
            "main.time.sleep", side_effect=clock.sleep
        ):
            main.stop_axis0_and_enter_idle(drive)

        velocities = [command[2] for command in drive.commands if command[0] == "velocity"]
        self.assertGreater(len(velocities), 2)
        self.assertEqual(velocities[0], -3.0)
        self.assertEqual(velocities[-1], 0.0)
        self.assertEqual(
            drive.commands[-2:],
            [("write", "axis0.requested_state", 1), ("system", "sc")],
        )

    def test_stop_axis_handles_positive_and_zero_speed(self):
        for initial_speed in (3.0, 0.0):
            with self.subTest(initial_speed=initial_speed):
                drive = FakeODrive()
                clock = FakeClock()
                drive.request_feedback = lambda axis: (
                    f"0 {max(0.0, initial_speed - clock.now * 5.0)}"
                )
                with patch("main.time.monotonic", side_effect=clock.monotonic), patch(
                    "main.time.sleep", side_effect=clock.sleep
                ):
                    main.stop_axis0_and_enter_idle(drive)
                velocities = [command[2] for command in drive.commands if command[0] == "velocity"]
                self.assertEqual(velocities[0], initial_speed)
                self.assertEqual(velocities[-1], 0.0)
                self.assertEqual(
                    drive.commands[-2:],
                    [("write", "axis0.requested_state", 1), ("system", "sc")],
                )
                self.assertTrue(all(a >= b >= 0 for a, b in zip(velocities, velocities[1:])))

    def test_stop_axis_rejects_invalid_feedback(self):
        for feedback in ("invalid", None, "0 nan", "inf 0", "0 1 2"):
            with self.subTest(feedback=feedback):
                drive = FakeODrive()
                drive.request_feedback = lambda axis: feedback
                with self.assertRaisesRegex(RuntimeError, "速度反馈无效"):
                    main.stop_axis0_and_enter_idle(drive)
                self.assertEqual(
                    drive.commands,
                    [
                        ("velocity", 0, 0, 0),
                        ("write", "axis0.requested_state", 1),
                        ("system", "sc"),
                    ],
                )

    def test_stop_axis_times_out_when_actual_speed_does_not_drop(self):
        drive = FakeODrive()
        clock = FakeClock()
        drive.request_feedback = lambda axis: "0 3"
        with patch("main.time.monotonic", side_effect=clock.monotonic), patch(
            "main.time.sleep", side_effect=clock.sleep
        ):
            with self.assertRaisesRegex(RuntimeError, "减速超时"):
                main.stop_axis0_and_enter_idle(drive)
        self.assertIn(("velocity", 0, 0, 0), drive.commands)
        self.assertEqual(
            drive.commands[-2:],
            [("write", "axis0.requested_state", 1), ("system", "sc")],
        )

    def test_stop_does_not_rearm_when_protection_occurs_during_ramp(self):
        for fault in ("axis0.motor.error", "axis0.current_state"):
            with self.subTest(fault=fault):
                drive = FakeODrive()
                clock = FakeClock()
                drive.request_feedback = lambda axis: "0 3"

                def trigger_fault(seconds):
                    clock.sleep(seconds)
                    drive.values[fault] = "1"

                with patch("main.time.monotonic", side_effect=clock.monotonic), patch(
                    "main.time.sleep", side_effect=trigger_fault
                ):
                    with self.assertRaises(RuntimeError):
                        main.stop_axis0_and_enter_idle(drive)
                self.assertEqual(drive.values["axis0.current_state"], "1")
                self.assertIn(("system", "sc"), drive.commands)
                self.assertNotIn(("write", "axis0.requested_state", 8), drive.commands)

    def test_stop_axis_rejects_failure_to_remain_closed_loop(self):
        drive = FakeODrive({"axis0.current_state": "1"})

        with self.assertRaisesRegex(RuntimeError, "STOP 前 AXIS0 已退出闭环"):
            main.stop_axis0_and_enter_idle(drive)

    @patch("main.time.sleep", return_value=None)
    def test_stop_clears_latched_motor_protection_while_idle(self, _sleep):
        drive = FakeODrive({"axis0.motor.error": "4096", "axis0.error": "64"})
        drive.request_feedback = lambda axis: "0 -15"

        with self.assertRaisesRegex(RuntimeError, "axis0.motor.error"):
            main.stop_axis0_and_enter_idle(drive)

        self.assertEqual(drive.values["axis0.current_state"], "1")
        self.assertEqual(drive.values["axis0.error"], "0")
        self.assertEqual(drive.values["axis0.motor.error"], "0")
        self.assertIn(("system", "sc"), drive.commands)
        self.assertNotIn(("write", "axis0.requested_state", 8), drive.commands)

        main.prepare_axis0_closed_loop(drive)
        self.assertEqual(drive.values["axis0.current_state"], "8")

    @patch("main.time.sleep", return_value=None)
    def test_prepare_rejects_latched_error_without_sending_velocity(self, _sleep):
        drive = FakeODrive({"axis0.motor.error": "4096"})
        with self.assertRaisesRegex(RuntimeError, "axis0.motor.error"):
            main.prepare_axis0_closed_loop(drive)
        self.assertEqual(drive.commands, [])

    @patch("main.set_balance_mode")
    def test_balance_command_enters_active_mode(self, set_balance_mode):
        should_stop = main.handle_operator_command("balance")

        self.assertFalse(should_stop)
        set_balance_mode.assert_called_once_with(main.BALANCE_MODE_ACTIVE)

    @patch("main.get_balance_status", return_value=(main.BALANCE_MODE_ACTIVE, 0.0))
    def test_stop_command_exits_after_balance(self, _get_balance_status):
        self.assertTrue(main.handle_operator_command("stop"))

    @patch("main.get_balance_status", return_value=("docked", 9.47))
    def test_stop_command_is_rejected_before_balance(self, _get_balance_status):
        self.assertFalse(main.handle_operator_command("stop"))

    def test_create_tuning_log_uses_logs_folder_and_expected_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "LOG_DIR", Path(directory) / "logs"):
                path, log_file, _writer = main.create_tuning_log()
                log_file.close()

            self.assertEqual(path.parent.name, "logs")
            header = path.read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertEqual(header.split(","), list(main.TUNING_LOG_FIELDS))

    def test_enqueue_tuning_log_never_waits_when_queue_is_full(self):
        log_queue = queue.Queue(maxsize=1)
        log_queue.put_nowait({"row": 1})

        self.assertFalse(main.enqueue_tuning_log({"row": 2}, log_queue))
        self.assertEqual(log_queue.get_nowait(), {"row": 1})

    def test_tuning_log_thread_batches_and_drains_rows_on_stop(self):
        row = {field: index for index, field in enumerate(main.TUNING_LOG_FIELDS)}
        log_queue = queue.Queue()
        log_queue.put(row)
        log_queue.put(row)
        stop_event = threading.Event()
        stop_event.set()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "LOG_DIR", Path(directory) / "logs"), patch.object(
                main, "stop_event", stop_event
            ), patch.object(main, "TUNING_LOG_BATCH_SIZE", 10):
                log_queue.put(main.TUNING_LOG_STOP)
                main.tuning_log_thread_function(log_queue)

            paths = list((Path(directory) / "logs").glob("*.csv"))
            self.assertEqual(len(paths), 1)
            with paths[0].open(newline="", encoding="utf-8-sig") as log_file:
                rows = list(csv.DictReader(log_file))
            self.assertEqual(len(rows), 2)

    @patch("main.get_tuning_snapshot")
    def test_build_tuning_log_row_adds_motor_current(self, get_snapshot):
        get_snapshot.return_value = {
            "pitch_deg": 1.0,
            "gyro_x_deg_s": 2.0,
            "motor_speed_rev_s": 3.0,
            "velocity_loop_output": 4.0,
            "angle_loop_output": 5.0,
            "raw_cascade_output_rev_s": 6.0,
            "limited_output_rev_s": 7.0,
        }

        row = main.build_tuning_log_row(
            {
                "motor_current_a": "8.5",
                "iq_setpoint_a": "9.0",
                "odrive_current_limit_a": "10.0",
                "odrive_input_vel_rev_s": "7.0",
                "odrive_state": "8",
                "odrive_axis_error": "0",
                "odrive_motor_error": "0",
                "odrive_controller_error": "0",
            },
            main.time.monotonic() - 1.0,
        )

        self.assertEqual(row["motor_current_a"], 8.5)
        self.assertEqual(row["iq_setpoint_a"], 9.0)
        self.assertEqual(row["odrive_current_limit_a"], 10.0)
        self.assertEqual(row["odrive_state"], 8)
        self.assertGreaterEqual(row["elapsed_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
