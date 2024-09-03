#include "balance.h"

CENTER_STRUCT Car;
PID_ERECT roll_pid;

// 左右平衡
float param_roll_Gyro[4]  = {100, 0, 0, 0};
float param_roll_Angle[4] = {100, 0, 0, 0};
float param_roll_Speed[4] = {0, 0, 0, 0};

char data[32];  // Buffer to hold the final string
char data1[] = "r axis0.encoder.vel_estimate";

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
float PID4_roll_gyro(PID_INFO *pid_info, float *PID_Parm, float NowPoint, float SetPoint, float filter_param)
{
    float output;

    // 1.计算误差
    pid_info->iError = (NowPoint - SetPoint);

    // 2.对误差进行一阶互补滤波,使得波形更加平滑，滤除高频干扰，防止速度突变。
    pid_info->iError = filter_param * pid_info->iError + (1 - filter_param) * pid_info->LastError;
    pid_info->SumError += pid_info->iError;

    // 3.积分限幅
    if (PID_Parm[3])
    {
        pid_info->SumError = lr_limit(pid_info->SumError, PID_Parm[3]);
    }

    // 4.计算输出
    output = PID_Parm[0] * pid_info->iError +
             PID_Parm[1] * pid_info->SumError +
             PID_Parm[2] * (pid_info->iError - pid_info->LastError);
    pid_info->LastError = pid_info->iError;

    return output;
}

float PID4_roll_angle(PID_INFO *pid_info, float *PID_Parm, float NowPoint, float SetPoint, float filter_param)
{
    float output;

    // 1.计算误差
    pid_info->iError = (NowPoint - SetPoint);

    // 2.对误差进行一阶互补滤波,使得波形更加平滑，滤除高频干扰，防止速度突变。
    pid_info->iError = filter_param * pid_info->iError + (1 - filter_param) * pid_info->LastError;
    pid_info->SumError += pid_info->iError;

    // 3.积分限幅
    if (PID_Parm[3])
    {
        pid_info->SumError = lr_limit(pid_info->SumError, PID_Parm[3]);
    }

    // 4.计算输出
    output = PID_Parm[0] * pid_info->iError +
             PID_Parm[1] * pid_info->SumError +
             PID_Parm[2] * (pid_info->iError - pid_info->LastError);
    pid_info->LastError = pid_info->iError;

    return output;
}

float PID4_roll_speed(PID_INFO *pid_info, float *PID_Parm, float NowPoint, float SetPoint, float filter_param)
{
    float output;

    // 1.计算误差
    pid_info->iError = (NowPoint - SetPoint);

    // 2.对误差进行一阶互补滤波,使得波形更加平滑，滤除高频干扰，防止速度突变。
    pid_info->iError = filter_param * pid_info->iError + (1 - filter_param) * pid_info->LastError;
    pid_info->SumError += pid_info->iError;

    // 3.积分限幅
    if (PID_Parm[3])
    {
        pid_info->SumError = lr_limit(pid_info->SumError, PID_Parm[3]);
    }

    // 4.计算输出
    output = PID_Parm[0] * pid_info->iError +
             PID_Parm[1] * pid_info->SumError +
             PID_Parm[2] * (pid_info->iError - pid_info->LastError);
    pid_info->LastError = pid_info->iError;

    return output;
}
/*****************---------PID参数初始化---------*****************/
void pid_param_init(PID_INFO *pid_info)
{
    pid_info->iError = 0;
    pid_info->SumError = 0;
    pid_info->PrevError = 0;
    pid_info->LastError = 0;
    pid_info->LastData = 0;
}
/*****************---------PID参数初始化---------*****************/
//-------------------------------------------------------------------------------------------------------------------
//  @brief      odrive发送指令
//  @note       t为0则是请求编码器速度指令
//              t不为0则是目标速度值
//
//  @author     LateRain
//  @date       2024/9/3
//-------------------------------------------------------------------------------------------------------------------
void OdriveCommand(UART_HandleTypeDef *huart, float t)
{
    int integer_part = (int)t;  // Extract the integer part of the float
    int decimal_part = (int)((t - integer_part) * 100);  // Extract the first two decimal digits

    // Manually construct the string "v 0 xx.yy\n"
    int index = 0;
    data[index++] = 'v';
    data[index++] = ' ';
    data[index++] = '0';
    data[index++] = ' ';

    // Convert the integer part to string
    if (integer_part < 10) data[index++] = '0' + integer_part;
    else
    {
        data[index++] = '0' + (integer_part / 10);  // Tens place
        data[index++] = '0' + (integer_part % 10);  // Units place
    }

    data[index++] = '.';

    // Convert the decimal part to string
    data[index++] = '0' + (decimal_part / 10);  // First decimal place
    data[index++] = '0' + (decimal_part % 10);  // Second decimal place

    data[index++] = '\n';
    data[index] = '\0';  // Null-terminate the string

    // Transmit the string via UART
    if(t != 0)
        HAL_UART_Transmit_IT(huart, (uint8_t*)data, strlen(data));
    else if(t == 0)
        HAL_UART_Transmit_IT(huart, (uint8_t*)data1, strlen(data1));
}

//-------------------------------------------------------------------------------------------------------------------
//  @brief      TIM1中断回调5ms
//  @param      PID_Parm        句柄
//  @return     output          void
//  @note       if(htim->Instance == TIM1)判断哪个定时器中断
//              记得在主函数用HAL_TIM_Base_Start_IT(&htim1);开启定时器
//  @author     LateRain
//  @date       2024/9/2
//-------------------------------------------------------------------------------------------------------------------
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{   
    if(htim->Instance == TIM1)
    {

        static uint8_t itrt_flag = 0;
        itrt_flag++;

        // 5ms角速度环
        if (-20 < imuData.angle.roll && imuData.angle.roll < 20)
        {
            Car.roll_Gyro_output = PID4_roll_gyro(&roll_pid.roll_gyro_pid, param_roll_Gyro, imuData.gyro.gyroX, 0, 0.5);
            Car.roll_Gyro_output = lr_limit(Car.roll_Gyro_output, 10000);
        }
        else // 倒地保护
            Car.roll_Gyro_output = 0;
        // 10ms角度环
        if(0 == (itrt_flag % 2))
        {
            Car.roll_Angle_output = PID4_roll_angle(&roll_pid.roll_angle_pid, param_roll_Angle, imuData.angle.roll, 0, 0.5);
            Car.roll_Angle_output = lr_limit(Car.roll_Angle_output, 500);
        }
        // 15ms速度环
        if(0 == (itrt_flag % 3))
        {
            itrt_flag = 0;
            OdriveCommand(&huart3, 0);
        }

        // OdriveCommand_send_target(&huart3, 1.52);


    }
}
