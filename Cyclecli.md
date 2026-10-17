# ODrive 命令行记录

- 创建时间：2026-09-29 10:40:18（UTC+08:00）
- 功能：通过 Windows PowerShell 连接 `COM27` 上的 ODrive，读取 Motor 0 的当前位置，以当前位置加 10 圈为目标，设置位置控制命令的速度上限为 4 圈/秒，并在 15 秒内轮询位置和速度反馈。

## 重要状态与安全提醒

**此命令曾在设备上实际运行，但电机没有运动。不要直接重跑。**

复现时脚本能读取位置，并发送目标位置；但轮询期间反馈始终为 `0`。之后只读检查发现驱动器参数为 `control_mode=3`（位置控制）、`input_mode=2`（速度斜坡模式）、`input_pos=10`。初步原因是位置目标命令与驱动器输入模式不匹配。该脚本没有修正或检查输入模式，也没有确认位置命令已被驱动器执行。

此外，最后已知驱动器中可能保留 `axis0.controller.input_pos=10`。设备当前状态未知；切换输入模式可能令遗留目标生效，造成意外运动。先确认当前轴状态、错误、控制/输入模式、目标值和编码器反馈，再决定如何配置和测试。

## 完整 PowerShell 命令（历史复现版本，勿直接重跑）

```powershell
$env:PYTHONPATH = "D:\06_Code\01_EquiCycle\EquiCycle\control_unit\src"
@'
import time
from odrive_protocol.odrive_protocol import ODriveAsciiProtocol

drive = ODriveAsciiProtocol("COM27", 460800)
try:
    state = int(drive.read_parameter("axis0.current_state"))
    error = int(drive.read_parameter("axis0.error"))
    input_mode = int(
        drive.read_parameter("axis0.controller.config.input_mode")
    )

    if state != 8 or error != 0 or input_mode != 3:
        raise RuntimeError(
            f"条件不满足，未发送运动命令：state={state}, "
            f"error={error}, input_mode={input_mode}"
        )

    position, velocity = map(float, drive.request_feedback(0).split()[:2])
    target = position + 10.0
    print(f"当前位置：{position:.3f} 圈；目标：{target:.3f} 圈")
    drive.motor_position(0, target, velocity_lim=4.0)

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        time.sleep(0.2)
        position, velocity = map(float, drive.request_feedback(0).split()[:2])
        print(f"位置：{position:.3f} 圈；速度：{velocity:.3f} 圈/秒")
        if abs(position - target) < 0.05 and abs(velocity) < 0.1:
            print("到达目标位置")
            break
    else:
        print("15 秒内未到位；停止继续测试并检查状态")
finally:
    drive.close()
'@ | python
```

## 相关工程代码

- 串口协议封装：`control_unit/src/odrive_protocol/odrive_protocol.py`
- 当前所需 COM 端口：`COM27`
- 上述脚本所用串口波特率：`460800`

## 避免在 PowerShell 粘贴长 here-string

PowerShell 的 PSReadLine 可能在粘贴很长的多行内容时发生控制台渲染异常。可改为在项目根目录运行下面的短命令，执行合并后的脚本：

```powershell
$env:PYTHONPATH = "$PWD\control_unit\src"
python .\control_unit\windows_runtime\odrive_position_sequence.py
```

该脚本会先检查轴状态和错误，将目标安全地设为当前位置，再检查并配置位置输入模式。只有输入 `MOVE` 后才会发送从当前位置转 10 圈的运动命令。运行前仍需确保电机及机构固定、运动范围安全；若 COM27 报拒绝访问，应先处理串口访问问题。
