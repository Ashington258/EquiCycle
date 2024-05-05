#include <stdio.h>

// 数据类型枚举
typedef enum
{
    ACCELEROMETER = 0x51,
    GYROSCOPE = 0x52,
    ANGLE = 0x53,
    MAGNETIC = 0x54
} DataType;

// 定义数据结构
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
    float voltage; // 电压，只在角速度输出中有意义
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

// 计算校验和
unsigned char calculateChecksum(unsigned char data[])
{
    unsigned char checksum = 0;
    for (int i = 0; i < 10; i++)
    {
        checksum += data[i];
    }
    return checksum;
}

// 解析加速度数据
int parseAccelData(unsigned char data[], AccelData *accelData)
{
    // 验证校验和
    if (calculateChecksum(data) != data[10] || data[1] != ACCELEROMETER)
    {
        printf("Checksum error or incorrect data type\n");
        return 0;
    }

    short accelX_raw = short((short(data[3]) << 8) | data[2]);
    accelData->accelX = (float)accelX_raw / 32768.0 * 16.0;

    short accelY_raw = short((short(data[5]) << 8) | data[4]);
    accelData->accelY = (float)accelY_raw / 32768.0 * 16.0;

    short accelZ_raw = short((short(data[7]) << 8) | data[6]);
    accelData->accelZ = (float)accelZ_raw / 32768.0 * 16.0;

    short temp_raw = short((short(data[9]) << 8) | data[8]);
    accelData->temperature = (float)temp_raw / 100.0;

    return 1;
}

// 解析角速度数据
int parseGyroData(unsigned char data[], GyroData *gyroData)
{
    // 验证校验和
    if (calculateChecksum(data) != data[10] || data[1] != GYROSCOPE)
    {
        printf("Checksum error or incorrect data type\n");
        return 0;
    }

    short gyroX_raw = short((short(data[3]) << 8) | data[2]);
    gyroData->gyroX = (float)gyroX_raw / 32768.0 * 2000.0;

    short gyroY_raw = short((short(data[5]) << 8) | data[4]);
    gyroData->gyroY = (float)gyroY_raw / 32768.0 * 2000.0;

    short gyroZ_raw = short((short(data[7]) << 8) | data[6]);
    gyroData->gyroZ = (float)gyroZ_raw / 32768.0 * 2000.0;

    // 电压，非蓝牙产品，该数据无效
    gyroData->voltage = 0.0; // 电压数据无效

    return 1;
}

// 解析角度数据
int parseAngleData(unsigned char data[], AngleData *angleData)
{
    // 验证校验和
    if (calculateChecksum(data) != data[10] || data[1] != ANGLE)
    {
        printf("Checksum error or incorrect data type\n");
        return 0;
    }

    short roll_raw = short((short(data[3]) << 8) | data[2]);
    angleData->roll = (float)roll_raw / 32768.0 * 180.0;

    short pitch_raw = short((short(data[5]) << 8) | data[4]);
    angleData->pitch = (float)pitch_raw / 32768.0 * 180.0;

    short yaw_raw = short((short(data[7]) << 8) | data[6]);
    angleData->yaw = (float)yaw_raw / 32768.0 * 180.0;

    angleData->version = (data[9] << 8) | data[8];

    return 1;
}

// 解析磁场数据
int parseMagData(unsigned char data[], MagData *magData)
{
    // 验证校验和
    if (calculateChecksum(data) != data[10] || data[1] != MAGNETIC)
    {
        printf("Checksum error or incorrect data type\n");
        return 0;
    }

    short magX_raw = short((short(data[3]) << 8) | data[2]);
    magData->magX = (float)magX_raw;

    short magY_raw = short((short(data[5]) << 8) | data[4]);
    magData->magY = (float)magY_raw;

    short magZ_raw = short((short(data[7]) << 8) | data[6]);
    magData->magZ = (float)magZ_raw;

    short temp_raw = short((short(data[9]) << 8) | data[8]);
    magData->temperature = (float)temp_raw / 100.0;

    return 1;
}

int main()
{
    unsigned char receivedData[] = {
        0x55, 0x51, 0x23, 0x00, 0x01, 0x00, 0xFD, 0x07, 0x0F, 0x0B, 0xE8,
        0x55, 0x52, 0xFB, 0xFF, 0xF8, 0xFF, 0xFC, 0xFF, 0x0F, 0x0B, 0xAD,
        0x55, 0x53, 0x73, 0xFF, 0xEB, 0xFE, 0x0E, 0x00, 0x8B, 0x46, 0xE2,
        0x55, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA9};

    AccelData accelData;
    GyroData gyroData;
    AngleData angleData;
    MagData magData;

    // 解析数据
    for (int i = 0; i < sizeof(receivedData) / 11; i++)
    {
        unsigned char *data = receivedData + i * 11;
        switch (data[1])
        {
        case ACCELEROMETER:
            if (parseAccelData(data, &accelData))
            {
                printf("Accelerometer X: %.2f g\n", accelData.accelX);
                printf("Accelerometer Y: %.2f g\n", accelData.accelY);
                printf("Accelerometer Z: %.2f g\n", accelData.accelZ);
                printf("Temperature: %.2f ℃\n", accelData.temperature);
            }
            break;
        case GYROSCOPE:
            if (parseGyroData(data, &gyroData))
            {
                printf("Gyroscope X: %.2f °/s\n", gyroData.gyroX);
                printf("Gyroscope Y: %.2f °/s\n", gyroData.gyroY);
                printf("Gyroscope Z: %.2f °/s\n", gyroData.gyroZ);
            }
            break;
        case ANGLE:
            if (parseAngleData(data, &angleData))
            {
                printf("Roll: %.2f °\n", angleData.roll);
                printf("Pitch: %.2f °\n", angleData.pitch);
                printf("Yaw: %.2f °\n", angleData.yaw);
                printf("Version: %d\n", angleData.version);
            }
            break;
        case MAGNETIC:
            if (parseMagData(data, &magData))
            {
                printf("Magnetometer X: %.2f\n", magData.magX);
                printf("Magnetometer Y: %.2f\n", magData.magY);
                printf("Magnetometer Z: %.2f\n", magData.magZ);
                printf("Temperature: %.2f ℃\n", magData.temperature);
            }
            break;
        default:
            printf("Unknown data type\n");
            break;
        }
    }

    return 0;
}
