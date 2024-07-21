#ifndef WIT_IMU_H
#define WIT_IMU_H

#include <stdint.h>

// 各个子数据结构定义
typedef struct
{
    float accelX;
    float accelY;
    float accelZ;
    float temperature;
} AccelData;

typedef struct
{
    float gyroX;
    float gyroY;
    float gyroZ;
    float voltage;
} GyroData;

typedef struct
{
    float roll;
    float pitch;
    float yaw;
    int version;
} AngleData;

typedef struct
{
    float magX;
    float magY;
    float magZ;
    float temperature;
} MagData;

// 总的数据结构定义
typedef struct
{
    AccelData accel;
    GyroData gyro;
    AngleData angle;
    MagData mag;
} IMUData;

// 全局变量声明
extern IMUData imuData;

// 函数声明
unsigned char calculateChecksum(unsigned char data[]);
int parseAccelData(unsigned char data[], AccelData *accelData);
int parseGyroData(unsigned char data[], GyroData *gyroData);
int parseAngleData(unsigned char data[], AngleData *angleData);
int parseMagData(unsigned char data[], MagData *magData);
int parseIMUData(unsigned char data[], IMUData *imuData);

#endif // WIT_IMU_H
