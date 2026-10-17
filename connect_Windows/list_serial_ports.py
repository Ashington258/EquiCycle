"""List serial ports visible to Windows."""

from serial.tools import list_ports


def main() -> None:
    ports = sorted(list_ports.comports(), key=lambda item: item.device)
    if not ports:
        print("No serial ports found.")
        return

    print("Available serial ports:")
    for port in ports:
        print(
            f"- {port.device}: {port.description} "
            f"[VID:PID={port.vid!s}:{port.pid!s}, serial={port.serial_number or '-'}]"
        )


if __name__ == "__main__":
    main()
