#include "odrive.h"

char command_send_v[33];  // Buffer to hold the final string
char command_get_v_0[] = "r axis0.encoder.vel_estimate\n";
char command_get_v_1[] = "r axis1.encoder.vel_estimate\n";
int flag_p;

//-------------------------------------------------------------------------------------------------------------------
//  @brief      odrive发送指令
//
//  @param      huart       串口句柄
//  @param      speed       期望速度，值域：-99.99~99.99，精度两位小数
//  @param      motor       电机选择，0动量轮电机，1后轮电机
//  @param      command     命令形式选择，0获取速度，1发送速度
//  @return     none
//  @note       使用示例：   OdriveCommand( &huart3, 11.11, 0, 1 );使用串口三给0电机发送11.11速度
//
//  @author     LateRain
//  @date       2024/9/3
//-------------------------------------------------------------------------------------------------------------------
void OdriveCommand(UART_HandleTypeDef *huart, float speed, int motor, int command)
{
    if( command ){
        // 判断速度正负
        if(speed < 0){
            speed = -speed;
            flag_p = 1;
        }

        // 分开速度的整数和小数
        int integer_part = (int)speed; 
        int decimal_part = (int)((speed - integer_part) * 100);

        // 指令帧头
        int index = 0;
        if(!motor){     // 0，动量轮电机
        command_send_v[index++] = 'v';
        command_send_v[index++] = ' ';
        command_send_v[index++] = '0';
        command_send_v[index++] = ' ';
        }
        if(motor){      // 1，后轮电机
        command_send_v[index++] = 'v';
        command_send_v[index++] = ' ';
        command_send_v[index++] = '1';
        command_send_v[index++] = ' ';
        }
        // 速度指令
        if(flag_p)
            command_send_v[index++] = '-';
        if (integer_part < 10) 
            command_send_v[index++] = '0' + integer_part;  // 只有一位整数
        else{
            command_send_v[index++] = '0' + (integer_part / 10);  // 整数部分十位
            command_send_v[index++] = '0' + (integer_part % 10);  // 整数部分个位
        }
        command_send_v[index++] = '.';
        command_send_v[index++] = '0' + (decimal_part / 10);  // 小数部分第一位
        command_send_v[index++] = '0' + (decimal_part % 10);  // 小数部分第二位
        command_send_v[index++] = '\n';   //帧尾
        command_send_v[index] = '\0';  // 字符串结束
    }
    // 串口发送
    if( !command && !motor )     // 0电机，获取速度
        HAL_UART_Transmit_IT(huart, (uint8_t*)command_get_v_0, strlen(command_get_v_0));
    else if( !command && motor )     // 1电机，获取速度
        HAL_UART_Transmit_IT(huart, (uint8_t*)command_get_v_1, strlen(command_get_v_1));
    else if( command )      // 发送速度
        HAL_UART_Transmit_IT(huart, (uint8_t*)command_send_v, strlen(command_send_v));


    flag_p = 0;
}
//-------------------------------------------------------------------------------------------------------------------
//  @brief      odrive数据解析
//
//  @param      received_data       接收数据数组
//  @return     none
//  @note       由于空闲中断会和其他串口冲突，不得已使用普通中断一个字节一个字节接收且使用的
//              odrive获取速度指令返回的数据没有校验信息只有帧尾，故只能用小数点强行判断合理性，望解决
//
//  @author     LateRain
//  @date       2024/9/3
//-------------------------------------------------------------------------------------------------------------------
void Odrivedata_handle(char *received_data) 
{
    static float temp_speed;
    // 数据校验
    if( (received_data[1]=='.'&&received_data[0]!='-')||    // x.xxx格式
        (received_data[2]=='.')||       // xx.xxx or -x.xxx格式
        (received_data[3]=='.'&&received_data[0]=='-')  // -xx.xxx格式
      )
        temp_speed = atof(received_data);
    if( -50<=temp_speed&&temp_speed<=50 )   // 速度在合理范围，赋值使用
        motor_contl.speed_realtime = temp_speed;  

}
