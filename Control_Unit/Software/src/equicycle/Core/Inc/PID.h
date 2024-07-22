#ifndef PID_H
#define PID_H

#include "main.h"

#define lr_limit(a,b)   ((a) < -(b) ? -(b) : ((a) > (b) ? (b) : (a)))

/*****************---------结构体---------*****************/
typedef struct
{
    float iError;           // 误差
    float LastError;        // 上次误差
    float PrevError;        // 上上次误差
    float LastData;         // 上次数据
    float iErrorHistory[5]; // 历史误差
    float SumError;         // 累计误差
} PID_INFO;

typedef struct
{
    PID_INFO roll_gyro_pid;     // 动量轮串级 角速度环
    PID_INFO roll_angle_pid;    // 动量轮串级 角度环
    PID_INFO roll_speed_pid;    // 动量轮串级 速度环

} PID_ERECT;

extern PID_ERECT roll_pid;         //PID结构体
/*****************---------结构体---------*****************/
/*****************---------PID参数---------*****************/
extern float param_roll_Gyro[4]; 
extern float param_roll_Angle[4]; 
extern float param_roll_Speed[4];
/*****************---------PID参数---------*****************/
/*****************---------函数---------*****************/
float PID4_roll_gyro( PID_INFO *pid_info , float * PID_Parm , float NowPoint , float SetPoint ,float filter_param);
float PID4_roll_angle( PID_INFO *pid_info , float * PID_Parm , float NowPoint , float SetPoint ,float filter_param);
float PID4_roll_speed( PID_INFO *pid_info , float * PID_Parm , float NowPoint , float SetPoint ,float filter_param);
/*****************---------函数---------*****************/
#endif