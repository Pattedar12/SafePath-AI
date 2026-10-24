#define TRIG_PIN D1
#define ECHO_PIN D2
#define IR_PIN D6
#define BUZZER_PIN D5
#define RAIN_PIN D7

float readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);

  if (duration == 0) {
    return -1;
  }

  return duration * 0.0343 / 2.0;
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(IR_PIN, INPUT);
  pinMode(RAIN_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);

  Serial.println("SafePath AI IoT Started");
}

void loop() {
  float distance = readDistanceCM();

  int irValue = digitalRead(IR_PIN);
  int rainValue = digitalRead(RAIN_PIN);

  bool buzzerOn = false;

  if (irValue == LOW) {
    buzzerOn = true;
  }
  else if (distance > 0 && distance <= 20) {
    buzzerOn = true;
  }
  else if (distance > 20 && distance <= 50) {
    buzzerOn = (millis() % 200 < 100);
  }
  else if (distance > 50 && distance <= 100) {
    buzzerOn = (millis() % 600 < 100);
  }
  else if (distance > 100 && distance <= 150) {
    buzzerOn = (millis() % 1200 < 100);
  }

  digitalWrite(BUZZER_PIN, buzzerOn ? HIGH : LOW);

  Serial.print("Distance: ");

  if (distance < 0) {
    Serial.print("INVALID");
  } else {
    Serial.print(distance);
    Serial.print(" cm");
  }

  Serial.print(" | IR: ");
  Serial.print(irValue);

  Serial.print(" | Rain: ");
  Serial.print(rainValue);

  Serial.print(" | Buzzer: ");
  Serial.println(buzzerOn ? "ON" : "OFF");

  delay(50);
}