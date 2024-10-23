#!/bin/bash

# 检查是否安装了 usbutils
if ! command -v lsusb &> /dev/null
then
    echo "lsusb 命令未找到，正在安装 usbutils..."
    sudo apt update
    sudo apt install -y usbutils
fi

echo "连接的 USB 设备信息："
echo "========================="

# 使用 lsusb 命令获取 USB 设备信息
lsusb | while read line; do
    # 获取设备的 Bus 和 Device 信息
    bus=$(echo $line | awk '{print \$2}')
    device=$(echo $line | awk '{print \$4}' | sed 's/:$//')

    # 获取设备的详细信息
    details=$(udevadm info --query=all --name=/dev/bus/usb/$bus/$device 2>/dev/null)

    # 输出设备名称和连接的端口
    echo "设备信息: $line"
    echo "$details"
    echo "-------------------------"
done

echo "========================="
echo "USB 设备信息获取完成。"
