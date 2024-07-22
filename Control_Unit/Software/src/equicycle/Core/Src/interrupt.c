#include "interrupt.h"

CENTER_STRUCT Car;

void itrt_5ms(void)
{
    
    static uint8_t itrt_flag = 0;
    itrt_flag++;

    // 5ms角速度环
    if( -20 < imuData.angle.roll && imuData.angle.roll <20 ){
        Car.roll_Gyro_output = PID4_roll_gyro( &roll_pid.roll_gyro_pid, param_roll_Gyro, imuData.gyro.gyroX, Car.roll_Angle_output, 0.5 );
        Car.roll_Gyro_output = lr_limit( Car.roll_Gyro_output, 10000 );
    }
    else        //倒地保护
        Car.roll_Gyro_output = 0;


    // 10ms角度环
    if (itrt_flag %= 2){
        Car.roll_Angle_output = PID4_roll_angle( &roll_pid.roll_angle_pid, param_roll_Angle, imuData.angle.roll, Car.roll_Speed_output, 0.5 );
        Car.roll_Angle_output = lr_limit( Car.roll_Angle_output, 500 );
    }

    // 15ms速度环
    if (itrt_flag %= 3){
        itrt_flag = 0;

        // Car.roll_Speed_output = PID4_roll_speed( &roll_pid.roll_speed_pid, param_roll_Speed, , , 0.5);
        // Car.roll_Speed_output = lr_limit( Car.roll_Speed_output, 20 );
    }
}
