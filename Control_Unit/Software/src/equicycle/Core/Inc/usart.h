/*
 * @Author: Ashington ashington258@proton.me
 * @Date: 2024-07-21 21:24:08
 * @LastEditors: Ashington ashington258@proton.me
 * @LastEditTime: 2024-07-23 08:11:09
 * @FilePath: \equicycle\Core\Inc\usart.h
 * @Description: 请填写简介
 * 联系方式:921488837@qq.com
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved.
 */
/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    usart.h
 * @brief   This file contains all the function prototypes for
 *          the usart.c file
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2024 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __USART_H__
#define __USART_H__

#ifdef __cplusplus
extern "C"
{
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

  /* USER CODE BEGIN Includes */

  /* USER CODE END Includes */

  extern UART_HandleTypeDef huart1;

  extern UART_HandleTypeDef huart2;

  /* USER CODE BEGIN Private defines */

  /* USER CODE END Private defines */

  void MX_USART1_UART_Init(void);
  void MX_USART2_UART_Init(void);

  /* USER CODE BEGIN Prototypes */
  void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart);
  void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size);
  void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart);
  /* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __USART_H__ */
