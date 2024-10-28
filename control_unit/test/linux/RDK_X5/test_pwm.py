#!/usr/bin/env python3
import sys
import signal

import Hobot.GPIO as GPIO
import time

def signal_handler(signal, frame):
    sys.exit(0)

# 支持PWM的管脚: 32 and 33, 在使用PWM时，必须确保该管脚没有被其他功能占用
output_pin = 33

GPIO.setwarnings(False)

def main():
    # Pin Setup:
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)
    # 支持的频率范围： 48KHz ~ 192MHz
    p = GPIO.PWM(output_pin, 48000)
    # 初始占空比 25%， 先每0.25秒增加5%占空比，达到100%之后再每0.25秒减少5%占空比
    val = 25
    incr = 5
    p.ChangeDutyCycle(val)
    p.start(val)

    print("PWM running. Press CTRL+C to exit.")
    try:
        while True:
            time.sleep(0.25)
            if val >= 100:
                incr = -incr
            if val <= 0:
                incr = -incr
            val += incr
            p.ChangeDutyCycle(val)
    finally:
        p.stop()
        GPIO.cleanup()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
