int IN1 = 3;
int IN2 = 4;
int IN3 = 5;
int IN4 = 6;
int ENA = 2; // Controla la velocidad del motor A
int ENB = 7; // Controla la velocidad del motor B

void setup() {
  Serial.begin(9600); // Inicia la comunicación serial a 9600 baudios

  // Configurar pines de motor como salida
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  Serial.println("Serial Communication On"); // Indicar que la comunicación serial está activa
}

void loop() {
  if (Serial.available() > 0) {  // Comprueba si hay datos entrantes en el puerto serial
    String comando = Serial.readStringUntil('\n'); // Lee el comando enviado por el puerto serial hasta un salto de línea
    Serial.println(comando); // Opcional: Imprime el comando recibido

    if (comando == "avanzar") {
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      analogWrite(ENA, 255); // Velocidad máxima
      analogWrite(ENB, 255); // Velocidad máxima
    }
    else if (comando == "derecha") {
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      analogWrite(ENA, 255); // Velocidad máxima
      analogWrite(ENB, 255); // Velocidad máxima
    }
    else if (comando == "izquierda") {
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      analogWrite(ENA, 255); // Velocidad máxima
      analogWrite(ENB, 255); // Velocidad máxima
    }
    else if (comando == "reposo") {
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      analogWrite(ENA, 0); // Motor apagado
      analogWrite(ENB, 0); // Motor apagado
    }
  }
}
