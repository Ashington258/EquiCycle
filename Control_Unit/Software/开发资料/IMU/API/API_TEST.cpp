#include <stdio.h>
#include <stdlib.h>

// 解析加速度数据
float parseAcceleration(unsigned char high, unsigned char low) {
    short combined = (high << 8) | low;
    return (float)combined / 32768.0 * 16.0; // 加速度范围在±16g内
}

// 解析温度数据
float parseTemperature(unsigned char high, unsigned char low) {
    short combined = (high << 8) | low;
    return (float)combined / 100.0; // 温度范围
}

// 处理接收到的数据
int* processData(unsigned char data[], int dataLength) {
    // 验证数据长度是否正确
    if (dataLength != 10) {
        printf("数据长度错误！\n");
        return NULL;
    }

    // 计算校验和
    int sum = 0;
    for (int i = 0; i < dataLength - 1; i++) {
        sum += data[i];
    }

    // 验证校验和
    if (sum != data[dataLength - 1]) {
        printf("校验和错误！\n");
        return NULL;
    }

    // 解析数据
    static int result[4];
    result[0] = parseAcceleration(data[3], data[2]);
    result[1] = parseAcceleration(data[5], data[4]);
    result[2] = parseAcceleration(data[7], data[6]);
    result[3] = parseTemperature(data[9], data[8]);

    return result;
}

int main() {
    unsigned char data[] = {0x55, 0x51, 0x23, 0x00, 0x01, 0x00, 0xFD, 0x07, 0x0F, 0x0B};
    int* result = processData(data, sizeof(data));

    if (result != NULL) {
        printf("加速度 X: %.3f g\n", (float)result[0]);
        printf("加速度 Y: %.3f g\n", (float)result[1]);
        printf("加速度 Z: %.3f g\n", (float)result[2]);
        printf("温度: %.2f ℃\n", (float)result[3]);
    }

    return 0;
}
