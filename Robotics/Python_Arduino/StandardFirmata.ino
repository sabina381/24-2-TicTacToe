// 이 파일은 ArduinoIDE에서 상단바
//  File >> Examples >> (Examples for Arduino Uno)Firmata >> StandardFirmata
// 로 들어가서 이곳에 스케치를 작성해야합니다.

#include "StepMotor.h"

Stepper stepper(8, 9, 10, 11);

void setup() {
    Serial.begin(9600); // 시리얼 통신 시작
    Serial.println("Setup complete");
}

void loop() {
    if (Serial.available()) {
        char command = Serial.read(); // Python에서 보낸 명령 읽기
        if (command == '1') {
            stepper.step(2000, 1); // 시계 방향으로 2000 스텝 이동
            Serial.println("Done: Clockwise"); // Python에 완료 신호 전송
        } else if (command == '0') {
            stepper.step(2000, -1); // 반시계 방향으로 2000 스텝 이동
            Serial.println("Done: Counterclockwise"); // Python에 완료 신호 전송
        }
    }
}
