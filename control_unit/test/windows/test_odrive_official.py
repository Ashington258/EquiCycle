import odrive

odrv0 = odrive.find_any()
print(str(odrv0.vbus_voltage))
