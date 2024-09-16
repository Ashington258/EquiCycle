#ifndef ODRIVE_H
#define ODRIVE_H

#include "main.h"

void OdriveCommand(UART_HandleTypeDef *huart, float speed, int motor, int command);
void Odrivedata_handle(char *received_data);

#endif