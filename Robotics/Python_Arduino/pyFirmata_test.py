import pyfirmata
import serial
import time

# 시리얼 포트 설정 (포트를 정확히 확인하세요)
ser = serial.Serial('/dev/cu.usbserial-110', 9600)
print("Communication Successfully started")

while True:
    command = input("Enter '1' for clockwise or '0' for counterclockwise: ")
    if command in ['1', '0']:
        ser.write(command.encode())  # 명령 전송
        print(f"Sent command: {command}")

        # Arduino로부터 완료 신호 대기
        while True:
            if ser.in_waiting > 0:  # 수신된 데이터가 있으면
                response = ser.readline().decode().strip()  # 데이터 읽기
                print(f"Arduino Response: {response}")
                break
    else:
        print("Invalid command")
    time.sleep(1)
