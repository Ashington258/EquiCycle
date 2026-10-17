# CH100 Windows 采集与上位机

本目录从仓库历史分支 `Control_Unit_beta` 中与 CH100 采集、ZeroMQ 发布和实时绘图相关的实现整理而来，并修复了历史脚本中的编码、端口硬编码、缓存无上限和绘图延迟问题。

## 历史来源

| 功能 | 历史文件 | 相关提交 |
|---|---|---|
| CH100 协议解析 | `src/ch100_protocol/hipnuc_serial_parser.py` | `f6b0cc6` |
| 串口采集与 ZMQ 发布 | `src/ch100_protocol/hipnuc_serial_parser_zmq.py` | `e337c64`、`207d855` |
| IMU 实时绘图 | `test/test_imu_plot.py` | `08975f1` |
| ZMQ 接收测试 | `test/test_zmq.py` | `a0ae65c` |
| Windows 串口配置 | `src/config.json` | `Control_Unit_beta` |

历史绘图提交明确标注过“数据绘图显示有延迟”。本目录保留其“采集进程 + 上位机进程”的设计，但对实现进行了可运行化整理。

## 文件说明

- `ch100_protocol.py`：CH100 HI91 帧解析和 CRC16 校验。
- `ch100_publisher.py`：读取 Windows COM 口，解析后通过 ZeroMQ 发布。
- `imu_plot.py`：上位机实时曲线，显示姿态角、角速度和加速度。
- `list_serial_ports.py`：列出 Windows 当前可用串口。
- `config.json`：串口、波特率、ZMQ 地址和显示参数。
- `requirements.txt`：独立运行所需依赖。

## 硬件连接

1. 使用支持数据传输的 USB-Micro USB 线连接 CH100 和电脑。
2. 在 Windows“设备管理器 → 端口 (COM 和 LPT)”中确认 CH100 的 COM 号。
3. 如果没有出现串口，先安装 CH100 所用 USB 转串口芯片的驱动。
4. 不要只使用充电线；充电线不会枚举串口设备。

## 安装

建议在项目虚拟环境中执行：

```powershell
python -m pip install -r connect_Windows\requirements.txt
```

## 确认串口

```powershell
python connect_Windows\list_serial_ports.py
```

然后修改 `config.json` 中的 `port`，例如：

```json
"port": "COM22"
```

CH100 历史配置使用 `460800` 波特率。如果设备输出配置不同，应同步修改 `baudrate`。

## 启动采集

在第一个终端运行：

```powershell
python connect_Windows\ch100_publisher.py
```

正常情况下会持续显示接收帧数、CRC 错误数以及最新的 Roll/Pitch/Yaw。可先转动 CH100，确认角度随姿态变化。

## 启动上位机

在第二个终端运行：

```powershell
python connect_Windows\imu_plot.py
```

上位机通过 `tcp://127.0.0.1:5555` 接收数据。采集端和上位机在同一台电脑时无需修改地址。

如果上位机在另一台电脑：

1. 采集端保持 `bind_address` 为 `tcp://*:5555`；
2. 将上位机 `connect_address` 改成采集电脑 IP，例如 `tcp://192.168.1.10:5555`；
3. 在 Windows 防火墙中允许 TCP 5555 端口。

## 首次验证顺序

1. `list_serial_ports.py` 能看到 CH100；
2. `ch100_publisher.py` 的有效帧计数持续增长；
3. 静止时角度和角速度基本稳定；
4. 分别绕三个轴转动，确认 Roll/Pitch/Yaw 和 Gyr X/Y/Z 的对应关系；
5. 再启动 `imu_plot.py` 查看实时曲线；
6. 在未确认坐标轴、正负方向、单位和零点前，不要把数据直接接入平衡电机闭环。

## 常见问题

- **打不开 COM 口**：关闭串口助手和其他占用该端口的程序。
- **一直没有有效帧**：检查 COM 号、波特率和 CH100 是否配置为输出 HI91 帧。
- **CRC 错误持续增加**：通常是波特率不匹配、线路不稳定或数据帧格式不同。
- **曲线不更新**：先确认采集端已运行，再检查 `bind_address` 与 `connect_address` 的端口是否一致。
- **数据显示过快或卡顿**：调小 `display.max_points` 或增大 `display.refresh_interval_ms`。
