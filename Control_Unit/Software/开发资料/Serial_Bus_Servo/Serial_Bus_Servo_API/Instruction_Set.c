/*
 * @Author: 小叶同学
 * @Date: 2024-03-20 10:16:35
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-03-20 12:39:09
 * @Description: 请填写简介
 */
// Control_Unit\Software\BSP\Serial_Bus_Servo

char *commands[] = {
    "#000P1500T1000!", // 1. 控制舵机
    "#000PVER!",       // 2. 读取舵机版本号
    "#000PID!",        // 3. 指定 ID 检测
    "#000PID001!",     // 4. 指定修改 ID
    "#000PULK!",       // 5. 释放舵机扭力
    "#000PULR!",       // 6. 恢复舵机扭力
    "#000PMOD!",       // 7. 读取舵机当前的工作模式
    "#000PMOD1!",      // 8. 设置舵机工作模式
    "#000PRAD!",       // 9. 读取舵机当前位置
    "#000PDPT!",       // 10.暂停舵机
    "#000PDCT!",       // 11. 继续舵机
    "#000PDST!",       // 12. 停止舵机
    "#000PBD0!",       // 13. 设置舵机通信波特率
    "#000PSCK!",       // 14. 纠正偏差
    "#000PCSD!",       // 15. 设置舵机启动位置
    "#000PCSM!",       // 16. 去除初始值
    "#000PCSR!",       // 17. 恢复初始值
    "#000PSMI!",       // 18. 设置舵机最小值
    "#000PSMX!",       // 19. 设置舵机最大值
    "#000PCLE0!",      // 20. 半恢复出厂设置
    "#000PCLE!",       // 21. 全恢复出厂设置
    "#000PRTV!"        // 22. 获取温度和电压
};

// while (1)
// {

//     /* USER CODE END WHILE */

//     /* USER CODE BEGIN 3 */
//     static uint8_t test = 6;
//     // 使能发送功能，每次发送前需要调用此函数
//     HAL_HalfDuplex_EnableTransmitter(&huart1);
//     HAL_UART_Transmit(&huart1, &test, 1, 2000);
//     // 使能接收功能。每次接收前需要调用此函数
//     HAL_HalfDuplex_EnableReceiver(&huart1);
//     HAL_UART_Receive(&huart1, &res, 1, 2000);

//     if (res == 6)
//     {
//         test++;
//     }
// }

// 主函数调用示例
#include "servo.h"

UART_HandleTypeDef huart1;
uint8_t buffer[100];

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();

    while (1)
    {
        servo_command(&huart1, "#000P1500T1000!"); // 控制舵机
        HAL_Delay(1000);
        receive_data(&huart1, buffer); // 接收并处理数据
        HAL_Delay(1000);
    }
}
