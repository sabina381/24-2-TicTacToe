// 이 파일은 ArduinoIDE에서 상단바
//  File >> Examples >> (Examples for Arduino Uno)Firmata >> StandardFirmata
// 로 들어가서 이곳에 스케치를 작성해야합니다.

#include "StepMotor.h"

Stepper stepper_x(8, 9, 10, 11);
Stepper stepper_y(3, 4, 5, 6);
Stepper stepper_z(A5, A4, A3, A2);

TicTacToeArtist artist(21, 21, &stepper_x, &stepper_y, &stepper_z);

int current_state = 0; // 현재 상태를 추적하는 변수

void setup() {
    Serial.begin(115200); // 시리얼 통신 시작
    Serial.println("Setup complete");
}

void loop() {
    if (Serial.available()) {
        char command = Serial.read(); // Python에서 보낸 명령 읽기

        switch (current_state) {
            case 0 :
                switch (command) {
                    case 'start' :
                        artist.drawGameBoard();
                        Serial.println("Done: draw Game board");
                        current_state = 0;
                        break;

                    case 'O' :
                        current_state = 1;
                        Serial.prinln("Setting O");
                        break;

                    case 'X' :
                        current_state = 2;
                        Serial.println("Setting X");
                        break;

                    default:
                        Serial.prinln("Nothing");
                }
                break;

            case 1 :
                int int_command = command.toInt();
                artist.drawCircle(int_command);
                Serial.println("Done: drawCircle");
                current_state = 0;
                break;
            
            case 2 :
                int int_command = command.toInt();
                artist.drawX(int_command);
                Serial.println("Done: drawX");
                current_state = 0;
                break;

            default:
                Serial.prinln("Nothing");
        }
    }
}
