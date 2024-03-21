/*
 * @Author: 小叶同学
 * @Date: 2024-03-20 20:06:37
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-03-20 20:11:20
 * @Description: 请填写简介
 */
#ifndef BSP_ENCODER_SERIAL_BUS_SERVO_H_
#define BSP_ENCODER_SERIAL_BUS_SERVO_H_

#include "stm32f1xx_hal.h"

void servo_command(UART_HandleTypeDef *huart, char *command);
void receive_data(UART_HandleTypeDef *huart, uint8_t *buffer);

#endif /* BSP_ENCODER_SERIAL_BUS_SERVO_H_ */
