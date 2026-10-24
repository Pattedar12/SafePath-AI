/* SafePath AI ESP32 buzzer serial receiver
   Upload using Arduino IDE. Serial baud rate must match the Streamlit app (115200).
   GPIO 23 is used for a low-current active buzzer module. Adapt wiring for your module.
*/
const int BUZZER_PIN = 23;
String commandBuffer = "";
bool alertActive = false;

void setBuzzer(bool on) {
  alertActive = on;
  digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
}

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  setBuzzer(false);
  Serial.begin(115200);
  Serial.println("SAFEPATH_ESP32_READY");
}

void loop() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      commandBuffer.trim();
      if (commandBuffer == "TEST") {
        setBuzzer(true);
        delay(350);
        setBuzzer(false);
        Serial.println("BUZZER_TEST_OK");
      } else if (commandBuffer == "ALERT_ON") {
        setBuzzer(true);
        Serial.println("ALERT_ACTIVE");
      } else if (commandBuffer == "ALERT_OFF") {
        setBuzzer(false);
        Serial.println("ALERT_CLEARED");
      } else if (commandBuffer.length() > 0) {
        Serial.println("UNKNOWN_COMMAND");
      }
      commandBuffer = "";
    } else if (commandBuffer.length() < 64) {
      commandBuffer += ch;
    }
  }
}
