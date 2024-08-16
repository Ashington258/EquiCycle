#include "balance.h"

CENTER_STRUCT Car;
PID_ERECT roll_pid;

//左右平衡
float param_roll_Gyro[4]     = {  0  ,  0  ,  0  ,  0  }; 
float param_roll_Angle[4]    = {  0  ,  0  ,  0  ,  0  }; 
float param_roll_Speed[4]    = {  0  ,  0  ,  0  ,  0  };




//-------------------------------------------------------------------------------------------------------------------
//  @brief      串级PID计算函数
//  @param      pid_info        PID结构体，误差等值缓存在其中
//  @param      PID_Parm        PID参数
//  @param      NowPoint        观测值
//  @param      SetPoint        期望值
//  @param      filter_param    一阶互补滤波系数，0-1之间，表示当前值占比，一般给0.5，不需要滤波就给1
//  @return     output          PID计算输出
//  @note       
//
//  @author     LateRain
//  @date       2024/7/22
//-------------------------------------------------------------------------------------------------------------------
float PID4_roll_gyro( PID_INFO *pid_info , float * PID_Parm , float NowPoint , float SetPoint ,float filter_param)
{
    float output;

    // 1.计算误差
    pid_info->iError = ( NowPoint - SetPoint );

    // 2.对误差进行一阶互补滤波,使得波形更加平滑，滤除高频干扰，防止速度突变。
    pid_info->iError =  filter_param * pid_info->iError + (1-filter_param) * pid_info->LastError;  
    pid_info->SumError += pid_info->iError;

    // 3.积分限幅
    if( PID_Parm[3] )
    {
        pid_info->SumError = lr_limit( pid_info->SumError , PID_Parm[3] );
    }

    // 4.计算输出
    output = PID_Parm[0] * pid_info->iError +
             PID_Parm[1] * pid_info->SumError +
             PID_Parm[2] * ( pid_info->iError - pid_info->LastError );
    pid_info->LastError = pid_info->iError;

    return output;
}

float PID4_roll_angle( PID_INFO *pid_info , float * PID_Parm , float NowPoint , float SetPoint ,float filter_param)
{
    float output;

    // 1.计算误差
    pid_info->iError = ( NowPoint - SetPoint );

    // 2.对误差进行一阶互补滤波,使得波形更加平滑，滤除高频干扰，防止速度突变。
    pid_info->iError =  filter_param * pid_info->iError + (1-filter_param) * pid_info->LastError;  
    pid_info->SumError += pid_info->iError;

    // 3.积分限幅
    if( PID_Parm[3] )
    {
        pid_info->SumError = lr_limit( pid_info->SumError , PID_Parm[3] );
    }

    // 4.计算输出
    output = PID_Parm[0] * pid_info->iError +
             PID_Parm[1] * pid_info->SumError +
             PID_Parm[2] * ( pid_info->iError - pid_info->LastError );
    pid_info->LastError = pid_info->iError;

    return output;
}

float PID4_roll_speed( PID_INFO *pid_info , float * PID_Parm , float NowPoint , float SetPoint ,float filter_param)
{
    float output;

    // 1.计算误差
    pid_info->iError = ( NowPoint - SetPoint );

    // 2.对误差进行一阶互补滤波,使得波形更加平滑，滤除高频干扰，防止速度突变。
    pid_info->iError =  filter_param * pid_info->iError + (1-filter_param) * pid_info->LastError;  
    pid_info->SumError += pid_info->iError;

    // 3.积分限幅
    if( PID_Parm[3] )
    {
        pid_info->SumError = lr_limit( pid_info->SumError , PID_Parm[3] );
    }

    // 4.计算输出
    output = PID_Parm[0] * pid_info->iError +
             PID_Parm[1] * pid_info->SumError +
             PID_Parm[2] * ( pid_info->iError - pid_info->LastError );
    pid_info->LastError = pid_info->iError;

    return output;
}
/*****************---------PID参数初始化---------*****************/
void pid_param_init( PID_INFO *pid_info )
{
    pid_info->iError = 0;
    pid_info->SumError = 0;
    pid_info->PrevError = 0;
    pid_info->LastError = 0;
    pid_info->LastData = 0;
}

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