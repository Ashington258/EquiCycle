from odrive_protocol.odrive_protocol import ODriveAsciiProtocol


def main():
    # Initialize ODrive with the appropriate serial port
    odrive = ODriveAsciiProtocol(port="COM10")

    # Example: Set motor position
    # odrive.motor_position(0, -2, velocity_lim=1, torque_lim=0.1)
    odrive.motor_velocity(0, 0)
    # Get motor feedback
    feedback = odrive.request_feedback(0)
    print(f"Motor feedback: {feedback}")

    # Close the connection
    odrive.close()


if __name__ == "__main__":
    main()
