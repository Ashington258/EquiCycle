#include "steer.h"
#include "tim.h"

#define STEER_MID_VALUE 570

void steer_contorl( int pwm )//正数向左（逆时针）
{
    static uint16_t output;

    pwm = lr_limit_ab( pwm, -300, 300 );
    output = lr_limit_ab( (STEER_MID_VALUE + pwm), 270, 870 );

    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, output);    // 变大向左转
}