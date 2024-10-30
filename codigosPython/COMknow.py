import serial

def test_port(port):
    try:
        with serial.Serial(port, 9600, timeout=1) as ser:
            print(f"Enviando 'AT' al {port}")
            ser.write(b'AT\r\n')  # Comando AT para verificar la comunicación
            response = ser.readline().decode().strip()
            print(f"Respuesta de {port}: {response}")
    except serial.SerialException as e:
        print(f"No se pudo abrir el puerto {port}: {e}")

if __name__ == "__main__":
    # Reemplaza estos nombres de puerto con los que quieras probar
    test_port('COM5')
    #test_port('COM5')
