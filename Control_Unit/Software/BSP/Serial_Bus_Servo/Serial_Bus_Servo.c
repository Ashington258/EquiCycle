/*
 * @Author: 小叶同学
 * @Date: 2024-03-20 20:06:30
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-03-20 20:06:34
 * @Description: 请填写简介
 */
#include "string.h"
#include "Serial_Bus_Servo"

void servo_command(UART_HandleTypeDef *huart, char *command)
{
    HAL_HalfDuplex_EnableTransmitter(huart);
    HAL_UART_Transmit(huart, (uint8_t *)command, strlen(command), HAL_MAX_DELAY);
    HAL_HalfDuplex_EnableReceiver(huart);
}

void receive_data(UART_HandleTypeDef *huart, uint8_t *buffer)
{
    if (HAL_UART_Receive(huart, buffer, sizeof(buffer), HAL_MAX_DELAY) == HAL_OK)
    {
        // 处理接收到的数据
    }
}
