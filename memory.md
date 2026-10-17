# Project Memory

这份文件用于给新会话提供精简的项目背景和最近进展，不保存完整聊天记录。

## 项目背景

- 项目：EquiCycle，平衡自行车相关工程。
- 控制单元主要代码位于 `control_unit/src/`；ODrive 串口协议封装位于 `control_unit/src/odrive_protocol/odrive_protocol.py`，桌面电机控制界面位于 `control_unit/src/motor_control_ui.py`。
- 用户希望使用 Windows 电脑通过串口手动控制 ODrive。

## 当前 ODrive 排查记录（2026-09-29）

- Windows 硬件串口：CH100=`COM26`，ODrive=`COM27`，均配置 `460800` baud。设备端口号可能变化，实际连接前核对设备管理器。
- 独立 Windows 运行副本位于 `control_unit/windows_runtime/`，不会覆盖 Jetson 源码 `control_unit/src/`。入口为 `src/main.py`，COM 配置位于 `src/config.json`，运行说明为 `README.md`。舵机/视觉 UDP listener 暂不启动。
- Windows 启动命令（在 Windows runtime 目录执行）：`.\.venv\Scripts\Activate.ps1`，然后 `python src\main.py`；输入 `ARM` 才启动真实硬件控制，Ctrl+C 退出并在正常关闭时向 ODrive 轴 0 发送零速度。Python 导入需从 Windows runtime 目录启动，或设置 `$env:PYTHONPATH = "$PWD\src"`；从仓库根目录直接运行 `odrive_position_sequence.py` 会因导入路径不对而报 `ModuleNotFoundError`。
- 当前初步 pitch 控制：CH100 pitch（度）直接映射到轴 0 速度，增益 `1° -> 1 rev/s`，限幅 ±25 rev/s；pitch 超过 ±7° 时目标置零；pitch 数据超过 0.25 秒未更新时目标置零。不是经验证的自行车平衡闭环。
- Windows 主程序终端每秒打印 pitch（deg）、轴 0 目标速度和 ODrive 编码器反馈速度（rev/s 与 RPM），并标记 `active`、`tilt-stop` 或 `stale`。两者含义不同：target 是软件命令，actual 是编码器反馈。
- 已为 Windows 副本增加 pitch 控制测试；最近验证为 8 项测试通过、Python 编译通过。验证只覆盖软件逻辑，未由代理直接连接硬件。
- CH100 独立采集脚本 `connect_Windows/ch100_publisher.py` 在 COM26/460800 下报告约 200 frames/s、CRC errors=0，pitch 读数当时稳定；早先 Windows 主程序只出现单次 CRC 错误，暂未发现 CH100 持续通信故障。
- 最近一次 ODrive 只读状态（在换电池供电之前）：
  - 母线约 24.06 V；AXIS0 `current_state=8`（closed loop），axis/motor/encoder error 均为 0，`control_mode=2`（速度控制），`input_mode=2`（速度斜坡），停止后的 input_vel 与速度反馈为 0。
  - AXIS1 `current_state=1`，`axis1.error=256`、`encoder.error=4`；本次所述电机连接 AXIS0，因此不要把 AXIS1 用作当前电机的健康证据或测试轴。
- `control_unit/windows_runtime/odrive_position_sequence.py` 使用位置控制（`control_mode=3`、`input_mode=3`）发送当前位置 +10 圈，限速 4 rev/s。用户实际输出显示从约 0.022 圈起动，到约 2.510 圈时速度变为 0，目标约 10.022 圈；用户 Ctrl+C 中断，未完成 10 圈。
- 用户确认更换的是同款电机、接在 AXIS0；更换前运行 pitch 控制时目标正负变化但实际速度为 0。用户判断旧台式电源给动量轮负载时可能触发限流/过流保护并令 ODrive 掉电，随后已改用电池并反馈“确实可以了”。不要关闭/绕过电源保护，不要盲目提高限流；优先核对电源、电池及保护规格，动态加载时重新读取状态。
- **安全注意：** 位置序列被 Ctrl+C 中断，脚本发送的目标 `10.022` 圈可能曾残留于驱动器；此后用户换电池且报告可用，但当前目标/状态没有新读数确认。重新启动运动脚本或切换控制模式前先只读确认 ODrive 在线、状态/错误、控制与输入模式、目标值和编码器反馈；车轮架空，急停可用。

## 用户偏好

- 用户对 Python 环境不熟悉，操作步骤应给出可直接复制到 Windows PowerShell 的完整命令，并说明预期输出。
- 解释硬件动作风险；不要在未确认的情况下自行发送电机运动命令或修改驱动器状态。
- Windows 运行操作说明需要明确当前目录及从仓库根目录运行时的路径差异。
- 仅在用户主动提出时，才把后续重要对话加入本文件；不要自动记录每次会话内容。

## 维护约定

- 记录稳定背景、当前结论、未解决问题和关键安全事项，不粘贴完整对话或重复日志。
- 在阶段性进展后更新对应条目，并标注日期；过时结论应替换或明确标为历史，避免与当前状态混淆。
