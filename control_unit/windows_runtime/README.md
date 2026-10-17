# Windows CH100 + ODrive runtime

This directory is an isolated Windows copy of the control-unit runtime. It does
not modify the Jetson runtime in `control_unit/src`.

## Configuration

Edit `src/config.json` if Windows assigns different COM ports:

- CH100: `COM26` at `460800` baud
- ODrive: `COM32` at `460800` baud
- ODrive axis 0 current limit: `90.0 A`

After `ARM`, the runtime writes `odrive.current_limit_a` to
`axis0.motor.config.current_lim` before requesting closed loop, then reads it
back. Startup is aborted if the readback differs by more than `0.01 A`. This is
a runtime setting; it is not saved to ODrive nonvolatile configuration. Use
`90 A` only when the motor, ODrive, wiring, battery, and protection system are
rated for it, and keep the emergency cutoff available.

The three cascaded PID loops are configured in
`src/balance_control/balance_control.py`, inside
`CascadedPIDController.__init__`: `velocity_pid` is the velocity outer loop,
`angle_pid` is the pitch-angle middle loop, and `gyro_pid` is the pitch-rate
inner loop. On startup, the program prints all three loops' gains and limits,
then checks both serial ports before showing the `ARM` prompt.
`MOTOR_SPEED_SLEW_RATE` is `4000.0 rev/s²`; each command is limited to the
measured speed plus or minus this rate times the control interval.

The servo and servo listener are not started in this runtime.

## Install and run

Open PowerShell in `control_unit\windows_runtime`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\main.py
```

The program checks both serial ports, then asks for the exact confirmation
`ARM`. After confirmation it rejects any axis, motor, or encoder error, verifies
that axis 0 is calibrated, selects velocity/passthrough control, and enters
closed loop. It starts in docked mode with the measured dock angle of `+9.47`
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

While running, the terminal reports the CH100 pitch angle, active pitch
setpoint, commanded velocity, encoder-measured velocity, and controller status
about once per second. Velocities are shown in both revolutions per second and
RPM.

Starting `main.py` also starts parameter-tuning capture automatically. Each run
creates `logs\balance_tuning_YYYYMMDD_HHMMSS.csv` containing the host time,
elapsed time, pitch angle, gyro X, encoder-measured motor speed, velocity-loop
output, angle-loop output, raw three-cascade PID output, output after slew-rate
and speed limiting, and measured motor current. The control thread only submits
completed rows with a non-blocking operation to a bounded queue. A dedicated
logging thread writes batches of 50 rows and flushes at least every 0.5 seconds,
then drains the queue during normal shutdown. If storage cannot keep up and the
queue reaches 4096 rows, new log rows are dropped and reported instead of
blocking control; observation and motor control continue.
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
