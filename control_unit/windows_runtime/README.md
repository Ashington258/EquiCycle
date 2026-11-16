# Windows CH100 + ODrive runtime

This directory is an isolated Windows copy of the control-unit runtime. It does
not modify the Jetson runtime in `control_unit/src`.

## Configuration

Edit `src/config.json` if Windows assigns different COM ports:

- CH100: `COM26` at `460800` baud
- ODrive: `COM33` at `460800` baud
- ODrive axis 0 hard limits: `30.0 A`, `1.0 Nm`, and `15.0 rev/s`

After `ARM`, the runtime writes and reads back the fixed limits
`axis0.motor.config.current_lim=30.0`, `axis0.motor.config.torque_lim=1.0`,
and `axis0.controller.config.vel_limit=15.0` before requesting closed loop.
Startup is aborted if any readback differs by more than `0.01` in its unit. These
are runtime safety limits and are not saved to ODrive nonvolatile configuration;
they must not be automatically increased.

The tested ODrive velocity-loop gains are `vel_gain=0.10` and
`vel_integrator_gain=0.08`. Anticogging and automatic closed-loop startup are
disabled. The incremental encoder is saved as pre-calibrated, keeps startup
index search enabled, and does not repeat offset calibration on every boot. The
recovery script `odrive_apply_previous_config.py` currently specifies `60 A`
and derives its torque limit from `60 * torque_constant`; it does not match the
runtime's `30.0 A` / `1.0 Nm` limits and should not be run without checking the
hardware and resolving that discrepancy. Do not restore the former `0.65` / `0.20` gains:
they produced a measured zero-speed oscillation of approximately ±0.35 rev/s
and ±6 A on the replacement ODrive.

The three cascaded PID loops are configured in
`src/balance_control/balance_control.py`, inside
`CascadedPIDController.__init__`: `velocity_pid` is the velocity outer loop,
`angle_pid` is the pitch-angle middle loop, and `gyro_pid` is the pitch-rate
inner loop. On startup, the program prints all three loops' gains and limits,
then checks both serial ports before showing the `ARM` prompt.
`MOTOR_SPEED_SLEW_RATE` is `4000.0 rev/s²`; each software command is limited to
the measured speed plus or minus this rate times the control interval. ODrive's
independent `vel_ramp_rate` safety limit is `60.0 rev/s²` and is applied when
`input_mode=2` (velocity ramp mode).

The motor UI reads the selected axis's ODrive current, torque and velocity-ramp
settings on connection and shows their last-read values separately from the
editable, safety-capped controls. Editing a control does not update the drive;
`Apply Limits` writes and verifies the settings, selects velocity/ramp mode,
and enters closed loop. The ramp only applies in velocity control
(`control_mode=2`, `input_mode=2`); the position slider selects position mode
and therefore does not use the velocity-ramp limit. `torque_lim` is a ceiling
in Nm, not a commanded torque; status shows estimated torque as measured
`Iq_measured * torque_constant`. Do not infer the active hardware setting from
an editable control clipped to the UI maximum. The historical UI in commit
`fa5f4ed` sent position and velocity commands without configuring these limits.
Verify the selected axis, errors, power and readback before motion tests.

The servo and servo listener are not started in this runtime.

## Install and run

Open PowerShell in `control_unit\windows_runtime`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\main.py
```

To use the standalone motor-control UI with live CH100 acceleration values and
a rolling three-axis plot, launch it from this directory instead:

```powershell
python src\motor_control_ui.py
```

The UI reads CH100 port and baud rate from `src\config.json`. It displays the
raw acceleration values supplied by the CH100 parser; verify the sensor's scale
before interpreting them as physical units. Close `main.py` and any other
program using either serial port before launching the UI, because serial ports
cannot be shared between these applications.

For a controlled interactive run, add `--auto-arm-balance` only after the
vehicle is secured and the emergency cutoff is ready. This skips the `ARM`
prompt and enters `BALANCE` automatically after ODrive closed-loop startup:

```powershell
python src\main.py --auto-arm-balance
```

For a non-interactive launch, use `--balance-duration` to hold BALANCE for a
specified number of seconds and then execute the normal safe STOP sequence:

```powershell
python src\main.py --auto-arm-balance --balance-duration 10
```

Alternatively, provide a command file. Commands are processed in order;
`WAIT <seconds>` delays the next command, and comments beginning with `#` are
ignored. For example, `commands.txt` can contain:

```text
# Hold balance for ten seconds
WAIT 10
STOP
```

Run it with:

```powershell
python src\main.py --auto-arm-balance --command-file .\commands.txt
```

The program checks both serial ports, then asks for the exact confirmation
`ARM`. After confirmation it rejects any axis, motor, or encoder error, verifies
that axis 0 is calibrated, selects velocity/ramp control with a verified
`vel_ramp_rate` of `60.0 rev/s²`, and enters closed loop. It starts in docked
mode with the measured dock angle of `+9.47`
degrees and holds ODrive axis 0 at zero speed. Enter `BALANCE` only after the
landing clamp is ready to release; this resets the controllers and switches the
pitch setpoint to `0.00` degrees. After entering `BALANCE`, enter `STOP` to end
the run normally. Shutdown first joins the worker threads to prevent competing
velocity commands, then reads actual motor speed and ramps its target toward
zero at 5 rev/s². It waits for measured speed to fall within 0.05 rev/s and
verifies that axis 0 remains in closed loop during deceleration, then requests
IDLE and confirms that state before closing the serial port.
It does not request closed loop again or clear protection errors. A healthy
stop permits another `main.py` run without power-cycling ODrive; a protection
fault or communication failure is reported instead of claiming success.
The ramp has a timeout of the initial speed divided by 5 rev/s² plus 5 seconds.
If feedback fails while the axis is still confirmed healthy and in closed loop,
it attempts a zero target as a fallback; this fallback is not a ramp.
`STOP` is rejected before `BALANCE`. Ctrl+C runs the same shutdown sequence,
not an instantaneous emergency stop. Keep the physical emergency power cutoff
available during operation. After a successful stop the axis is IDLE, so active
holding torque is no longer available; mechanically support the vehicle.

If a subsequent launch reports `axis0.error=64` and
`axis0.motor.error=16777216`, the legacy firmware fields indicate MOTOR_FAILED
(0x40) and SYSTEM_LEVEL (0x01000000), respectively, not a watchdog fault.
Startup deliberately refuses to arm while these faults remain latched. On an
axis fault, the runtime also attempts to report the top-level ODrive `error`
in decimal and hexadecimal; an unavailable diagnostic does not hide the
original axis fault. The system-level protection's physical trigger still
requires matching firmware diagnostics. Leaving the previous run in closed
loop was a shutdown lifecycle defect, but does not by itself prove the cause
of SYSTEM_LEVEL. Investigate the protection trigger before clearing errors;
this runtime does not bypass protection or automatically clear faults.

Fault reports decode the legacy axis, motor, controller, and encoder bitmasks
and capture the current state, DC bus voltage, current limit and margin, and
requested/measured q-axis current. For example, motor error `0x10001000` is
`CURRENT_LIMIT_VIOLATION|UNKNOWN_TORQUE`; it is not proof that the motor merely
failed to produce requested torque. Check supply droop, phase wiring, motor
configuration, mechanical load, and the captured current values before
clearing the fault. Do not increase current limits as a substitute for finding
the protection trigger.

While running, the terminal reports the CH100 pitch angle, active pitch
setpoint, commanded velocity, encoder-measured velocity, and controller status
about once per second. Velocities are shown in both revolutions per second and
RPM. While the program is waiting for operator input, use `PID?` to view gains
or `PID <velocity|angle|gyro> <kp> <ki> <kd>` to hot-update one loop. Each
update resets that loop's integral and derivative history. These changes are
runtime-only and are not saved to ODrive or files.

Starting `main.py` also starts parameter-tuning capture automatically. Each run
creates `logs\balance_tuning_YYYYMMDD_HHMMSS.csv` containing the host time,
elapsed time, pitch angle, gyro X, encoder-measured motor speed, velocity-loop
output, angle-loop output, raw three-cascade PID output, output after slew-rate
and speed limiting, and measured motor current. The control thread writes each
completed row directly to the CSV file and flushes it immediately. ODrive feedback
reads use a 10 ms monotonic-clock cadence; missed periods are skipped rather than
replayed in a burst. Disk or serial latency can still cause missed periods.
Motor current is read from
`axis0.motor.current_control.Iq_measured`; close other ODrive serial clients
before startup. The capture also records `Iq_setpoint`, the configured current
limit, ODrive's received velocity target, current axis state, and
axis/motor/controller error codes. Static current limit is read on the first
successful feedback iteration; failed reads are logged and retried.
State and error diagnostics are refreshed every 0.25 seconds to limit serial
overhead. Additional reads can increase the feedback interval; CSV values are
cached measurements, not synchronized samples. The balance command slew rate
is 200 rev/s^2, so a 20 ms control interval permits a target at most 4 rev/s
away from measured motor speed before the absolute 15 rev/s speed limit is
applied. This bound uses measured speed, not the previous command, and does
not guarantee the same mechanical acceleration.

`main.py` also publishes each CH100 frame on TCP port 5555, so the real-time
plot can run without opening the CH100 serial port a second time. Keep
`main.py` running, open a second PowerShell at the repository root, and run:

```powershell
& .\control_unit\windows_runtime\.venv\Scripts\python.exe .\connect_Windows\imu_plot.py
```

Do not run `connect_Windows\ch100_publisher.py` at the same time as `main.py`;
both programs would try to open the same CH100 COM port.

## Read ODrive FET temperature

Connect and power the ODrive over USB, close `odrivetool` and other programs
using it, then run this read-only command from this directory:

```powershell
& .\.venv\Scripts\python.exe .\odrive_temperature.py
```

Example output:

```text
[2026-10-14 20:53:03] ODrive AXIS0 | FET temperature=36.3 °C
```

Add `--watch` to print once per second, or `--axis 1` to read AXIS1. The script
uses the firmware property `axisN.motor.fet_thermistor.temperature` through the
ODrive USB API and does not send motion or configuration commands.

## Motor/IMU interference test

To diagnose whether motor operation affects CH100 attitude measurements, close
`main.py` and any other program using either serial port, then run from this
directory:

```powershell
& .\.venv\Scripts\python.exe .\motor_imu_interference_test.py
```

The script requires the exact confirmation `RUN TEST`. It verifies that ODrive
AXIS0 is healthy, already in velocity control (`control_mode=2`, `input_mode=1`
or `2`), and initially nearly stopped. It then commands these stages:

1. 0 rev/s for 10 seconds
2. 5 rev/s for 10 seconds
3. 10 rev/s for 10 seconds
4. 15 rev/s for 10 seconds
5. 20 rev/s for 10 seconds
6. 25 rev/s for 10 seconds
7. 0 rev/s for 20 seconds

Every CH100 frame is saved under `logs\motor_imu_test_YYYYMMDD_HHMMSS.csv` with
the stage, target and encoder-measured motor speeds, roll/pitch/yaw, three-axis
gyroscope, and three-axis acceleration values. The script also publishes these
frames on `tcp://127.0.0.1:5555`, so `connect_Windows\imu_plot.py` can display all
three groups while the test runs. Press Ctrl+C to abort; normal and error
shutdown both command AXIS0 to 0 rev/s. The script does not change ODrive
control modes or clear errors.

## Safety

This program sends real velocity commands; it is not a hardware-free simulator.
Before entering `ARM`, raise the driven wheel, keep people clear, and have the
power cutoff/estop ready. Verify the ODrive errors, control/input modes, and
stored targets before starting. Startup selects velocity/passthrough modes;
the program does not calibrate the drive, clear errors, or save configuration. The controller uses the existing three cascaded loops
(wheel velocity, pitch angle, and pitch angular velocity), caps axis 0 at 15
turns/second, stops if pitch differs from the active setpoint by more than 12
degrees, and sends zero if CH100 data becomes stale for more than 0.25 seconds.
The existing PID gains are initial values only: first verify the correction
direction with the wheel raised, then tune on hardware before a free-standing
attempt. Based on the loaded IMU interference test, balance commands are limited
to 15 turns/second; measurements above that speed were not reliable enough for
closed-loop balance.

## Run the control tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```
