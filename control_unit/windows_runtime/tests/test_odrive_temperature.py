import unittest
from types import SimpleNamespace
from unittest.mock import patch

import odrive_temperature


class ODriveTemperatureTests(unittest.TestCase):
    def test_reads_requested_axis_fet_temperature(self):
        device = SimpleNamespace(
            axis1=SimpleNamespace(
                motor=SimpleNamespace(
                    fet_thermistor=SimpleNamespace(temperature=42.75)
                )
            )
        )

        self.assertEqual(
            odrive_temperature.read_fet_temperature(device, 1),
            42.75,
        )

    def test_formats_temperature_with_axis_unit_and_timestamp(self):
        output = odrive_temperature.format_temperature(
            0,
            36.26,
            timestamp="2026-10-14 20:53:03",
        )

        self.assertEqual(
            output,
            "[2026-10-14 20:53:03] ODrive AXIS0 | "
            "FET temperature=36.3 °C",
        )

    def test_uses_current_timestamp_when_not_supplied(self):
        with patch.object(
            odrive_temperature.time,
            "strftime",
            return_value="2026-10-14 20:53:03",
        ):
            output = odrive_temperature.format_temperature(0, 25.0)

        self.assertTrue(output.startswith("[2026-10-14 20:53:03]"))


if __name__ == "__main__":
    unittest.main()
