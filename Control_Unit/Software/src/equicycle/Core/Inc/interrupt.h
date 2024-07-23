#ifndef INTERRUPT_H
#define INTERRUPT_H

#include "main.h"

/*！！！！！！！！！
该文件不允许外传
代码是电气前几届学长写的
lr只是优化一下
涉及实验室纠纷，尽量避免外传
！！！！！！！！！*/

/*****************---------结构体---------*****************/
typedef struct
{
    // 串级PID
    float roll_Gyro_output;  // 左右平衡环输出飞轮占空比
    float roll_Angle_output; // 转向环输出飞轮占空比
    float roll_Speed_output; // 前后平衡环输出飞轮占空比

} CENTER_STRUCT;

extern CENTER_STRUCT Car;
/*****************---------结构体---------*****************/





#endif