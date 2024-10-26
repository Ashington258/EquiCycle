import Jetson.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
for pin in range(1, 41):
    try:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 100)
        print(f"Pin {pin} supports PWM.")
        pwm.stop()
    except Exception as e:
        print(f"Pin {pin} does not support PWM: {e}")

GPIO.cleanup()
