import serial
import time

bluetooth_port = 'COM4'
baud_rate = 9600

ser = serial.Serial(bluetooth_port, baud_rate, timeout=1)

def send_command(command):
    ser.write(command.to_bytes(1, 'big'))  # Envía el comando c1
    # omo un byte

def main():
    try:
        # Ejemplo de envío de comandos
        while True:
            command = input("Enter command (0=rest, 2=forward, 1=right, 3=left): ")
            if command.isdigit() and 0 <= int(command) <= 3:
                send_command(int(command))
            else:
                print("Invalid command. Please enter a number between 0 and 3.")
            time.sleep(1)
    finally:
        ser.close()

if __name__ == '__main__':
    main()
