/*
 * @Author: Ashington ashington258@proton.me
 * @Date: 2024-05-28 10:55:37
 * @LastEditors: Ashington ashington258@proton.me
 * @LastEditTime: 2024-07-22 21:58:50
 * @FilePath: \equicycle\api\vofa_protocol\vofa_protocol.c
 * @Description: 请填写简介
 * 联系方式:921488837@qq.com
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved.
 */

#include "vofa_protocol.h"
#include "main.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
// 尾帧的定义
static unsigned char tail[4] = {0x00, 0x00, 0x80, 0x7f};

// 将浮点数转换为小端字节流，并存储到指定的数组中
static void float_to_bytes(float value, unsigned char *bytes)
{
    unsigned char *p = (unsigned char *)&value;
    for (int i = 0; i < 4; i++)
    {
        bytes[i] = p[i];
    }
}

// 将多个浮点数转换为字节流并添加尾帧
static void create_byte_stream(const float *values, int num_values, unsigned char *byte_stream)
{
    for (int i = 0; i < num_values; i++)
    {
        float_to_bytes(values[i], &byte_stream[i * 4]);
    }
    // 添加尾帧
    memcpy(&byte_stream[num_values * 4], tail, 4);
}

// 单个浮点数的例子
void vofa_one(float A)
{
    unsigned char byte_stream[8]; // 4 bytes for float + 4 bytes for tail
    create_byte_stream(&A, 1, byte_stream);
    // 这里可以添加处理字节流的代码，例如发送或保存
    for (int i = 0; i < 8; i++)
    {
        printf("%02x ", byte_stream[i]);
    }
    printf("\n");
}

// 两个浮点数的例子
void vofa_two(float A, float B)
{
    float values[2] = {A, B};
    unsigned char byte_stream[12]; // 8 bytes for floats + 4 bytes for tail
    create_byte_stream(values, 2, byte_stream);
    // 这里可以添加处理字节流的代码，例如发送或保存
    for (int i = 0; i < 12; i++)
    {
        printf("%02x ", byte_stream[i]);
    }
    printf("\n");
}

// 三个浮点数的例子
void vofa_three(float A, float B, float C)
{
    float values[3] = {A, B, C};
    unsigned char byte_stream[16]; // 12 bytes for floats + 4 bytes for tail
    create_byte_stream(values, 3, byte_stream);
    // 这里可以添加处理字节流的代码，例如发送或保存
    HAL_UART_Transmit(&huart1, byte_stream, 16,10);
    for (int i = 0; i < 16; i++)
    {
        printf("%02x ", byte_stream[i]);
    }
    printf("\n");
}

// 四个浮点数的例子
void vofa_four(float A, float B, float C, float D)
{
    float values[4] = {A, B, C, D};
    unsigned char byte_stream[20]; // 16 bytes for floats + 4 bytes for tail
    create_byte_stream(values, 4, byte_stream);
    // 这里可以添加处理字节流的代码，例如发送或保存
    for (int i = 0; i < 20; i++)
    {
        printf("%02x ", byte_stream[i]);
    }
    printf("\n");
}

// 五个浮点数的例子
void vofa_five(float A, float B, float C, float D, float E)
{
    float values[5] = {A, B, C, D, E};
    unsigned char byte_stream[24]; // 20 bytes for floats + 4 bytes for tail
    create_byte_stream(values, 5, byte_stream);
    // 这里可以添加处理字节流的代码，例如发送或保存
    for (int i = 0; i < 24; i++)
    {
        printf("%02x ", byte_stream[i]);
    }
    printf("\n");
}

/// @brief
/// @param num_floats
/// @param float_array
/// @return
// BUG 程序发送的时候选择不同通道数的情况可能出现将内存区域其余无用地址发送出来，BUG有待修复
unsigned char *createFloatByteStream(int num_floats, float *float_array)
{
    // 尾帧
    unsigned char tail[4] = {0x00, 0x00, 0x80, 0x7f};

    // 分配内存：每个浮点数4字节，加上尾帧4字节
    unsigned char *byte_stream = (unsigned char *)calloc(num_floats * 4 + 4, sizeof(unsigned char));
    if (byte_stream == NULL)
    {
        perror("Memory allocation failed");
        return NULL;
    }

    // 将浮点数转换为字节流
    for (int i = 0; i < num_floats; i++)
    {
        // 小端方式拷贝浮点数
        memcpy(byte_stream + i * 4, (unsigned char *)(float_array + i), 4);
    }

    // 添加尾帧到字节流末尾
    memcpy(byte_stream + num_floats * 4, tail, 4);

    return byte_stream;
}
