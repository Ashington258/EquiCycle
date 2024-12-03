
要将 `/dev/ttyUSB0` 和 `/dev/ttyACM0` 设备与相应的 USB 设备 ID（例如 `10c4:ea60` 和 `1209:0d32`）对应起来，可以使用以下步骤：

### 步骤 1: 确认设备

首先，确保你已经插入了 USB 设备，并使用 `lsusb` 命令确认设备 ID：

```bash
lsusb
```

这将列出所有连接的 USB 设备及其 ID。

### 步骤 2: 使用 `dmesg` 查看设备信息

插入 USB 设备后，使用 `dmesg` 查看设备相关信息：

```bash
dmesg | grep tty
```

这将显示设备连接时的相关信息，包括设备节点（如 `/dev/ttyUSB0` 或 `/dev/ttyACM0`）和设备的 USB ID。

### 步骤 3: 识别设备与串口的对应关系

1. **查看 `dmesg` 输出**：查找在插入设备后产生的输出，通常会显示类似以下格式的信息：

   ```
   [    3.123456] usb 1-1.2: cp210x converter now attached to ttyUSB0
   [    3.123456] usb 1-1.3: pl2303 converter now attached to ttyUSB1
   ```

   这里可以看到 `cp210x` 设备（对应于 `10c4:ea60`）被分配到 `/dev/ttyUSB0`。

2. **使用 `udevadm` 查看设备信息**：你可以使用 `udevadm` 命令来获取更详细的信息：

   ```bash
   udevadm info --name=/dev/ttyUSB0 --attribute-walk
   ```

   这个命令将显示与 `/dev/ttyUSB0` 相关的所有属性，包括 USB ID。

### 步骤 4: 结合信息

通过上述步骤，你可以将设备 ID 与 `/dev/ttyUSB0` 和 `/dev/ttyACM0` 进行对应。例如：

- 如果你看到 `ttyUSB0` 显示的信息中包含 `10c4:ea60`，那么 `/dev/ttyUSB0` 对应于 `CP2102N USB to UART Bridge Controller`。
- 如果 `ttyACM0` 显示的信息中包含 `1209:0d32`，那么 `/dev/ttyACM0` 对应于 `ODrive 3.6 CDC Interface`。

### 示例

假设你在 `dmesg` 输出中看到：

```
[    3.123456] usb 1-1.2: new full-speed USB device number 4 using xhci_hcd
[    3.123456] usb 1-1.2: New USB device found, idVendor=10c4, idProduct=ea60
[    3.123456] usb 1-1.2: cp210x converter now attached to ttyUSB0
```

这表明 `10c4:ea60` 的设备被分配给了 `/dev/ttyUSB0`。

### 总结

通过 `dmesg` 和 `udevadm` 命令，你可以有效地将 `/dev/ttyUSB0` 和 `/dev/ttyACM0` 与相应的 USB 设备 ID 进行对应。如果你有其他问题或需要进一步的帮助，请告诉我！