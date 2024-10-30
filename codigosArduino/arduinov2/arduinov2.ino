#include <SoftwareSerial.h>

SoftwareSerial BTSerial(0, 1); // RX, TX

int IN1 = 3;
int IN2 = 4;
int IN3 = 5;
int IN4 = 6;
int ENA = 2;
int ENB = 7;

void setup() {
  Serial.begin(9600);
  BTSerial.begin(9600);
  Serial.println("Bluetooth On");

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
}

void loop() {
  if (BTSerial.available()) {
    int command = BTSerial.read();
    switch (command) {
      case 0:  // Reposo
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, LOW);
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, LOW);
        analogWrite(ENA, 0);
        analogWrite(ENB, 0);
        break;
      case 1:  // Avanzar
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
        digitalWrite(IN3, HIGH);
        digitalWrite(IN4, LOW);
        analogWrite(ENA, 128);
        analogWrite(ENB, 128);
        break;
      case 2:  // Derecha
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, HIGH);
        analogWrite(ENA, 128);
        analogWrite(ENB, 128);
        break;
      case 3:  // Izquierda
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, HIGH);
        digitalWrite(IN3, HIGH);
        digitalWrite(IN4, LOW);
        analogWrite(ENA, 128);
        analogWrite(ENB, 128);
        break;
      default:
        Serial.println("Unknown command");
    }
  }
}
