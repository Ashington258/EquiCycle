#ifndef VOFA_PROTOCOL_H
#define VOFA_PROTOCOL_H

void vofa_one(float A);
void vofa_two(float A, float B);
void vofa_three(float A, float B, float C);
void vofa_four(float A, float B, float C, float D);
void vofa_five(float A, float B, float C, float D, float E);
// 该函数用于自动识别通道数，但是似乎有一点问题，慎用，BUG还未修复
unsigned char *createFloatByteStream(int num_floats, float *float_array);

#endif // VOFA_PROTOCOL_H
