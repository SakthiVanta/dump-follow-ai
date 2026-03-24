/**
 * Tiny AI Robot — ESP32 Motor Firmware
 * =====================================
 * Hardware: L298N Motor Driver + 2 DC Motors
 *
 * Protocol (UART 115200 baud, newline-terminated ASCII):
 *   Receive: "M,<left>,<right>\n"   — set motor speeds [-200..200]
 *   Receive: "PING\n"               — heartbeat check
 *   Send:    "OK\n"                 — command accepted
 *   Send:    "ERR,<reason>\n"       — command rejected
 *
 * Pin assignments (L298N):
 *   IN1/IN2 → Left motor direction
 *   IN3/IN4 → Right motor direction
 *   ENA     → Left motor PWM enable
 *   ENB     → Right motor PWM enable
 */

// ─── Pin Definitions ──────────────────────────────────────────────────────────
#define IN1 26   // Left motor forward
#define IN2 27   // Left motor backward
#define IN3 14   // Right motor forward
#define IN4 12   // Right motor backward
#define ENA 25   // Left motor PWM (LEDC channel 0)
#define ENB 13   // Right motor PWM (LEDC channel 1)

// ─── PWM Config ───────────────────────────────────────────────────────────────
#define PWM_FREQ       1000
#define PWM_RESOLUTION 8      // 8-bit: 0–255
#define PWM_CH_LEFT    0
#define PWM_CH_RIGHT   1

// ─── Safety ───────────────────────────────────────────────────────────────────
#define WATCHDOG_TIMEOUT_MS 2000   // Stop motors if no command received
#define BAUD_RATE           115200
#define MAX_SPEED           200
#define CMD_BUFFER_SIZE     32

// ─── Globals ──────────────────────────────────────────────────────────────────
char cmdBuffer[CMD_BUFFER_SIZE];
uint8_t bufIndex = 0;
unsigned long lastCmdMs = 0;
bool watchdogArmed = false;

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);

  // Direction pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // PWM via LEDC (ESP32 native PWM)
  ledcSetup(PWM_CH_LEFT,  PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PWM_CH_RIGHT, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(ENA, PWM_CH_LEFT);
  ledcAttachPin(ENB, PWM_CH_RIGHT);

  stopMotors();

  Serial.println("READY");
}

// ─── Main Loop ────────────────────────────────────────────────────────────────
void loop() {
  // Read incoming bytes
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufIndex > 0) {
        cmdBuffer[bufIndex] = '\0';
        processCommand(cmdBuffer);
        bufIndex = 0;
      }
    } else if (bufIndex < CMD_BUFFER_SIZE - 1) {
      cmdBuffer[bufIndex++] = c;
    }
  }

  // Watchdog: stop if no command for WATCHDOG_TIMEOUT_MS
  if (watchdogArmed && (millis() - lastCmdMs > WATCHDOG_TIMEOUT_MS)) {
    stopMotors();
    watchdogArmed = false;
    Serial.println("WATCHDOG");
  }
}

// ─── Command Parser ───────────────────────────────────────────────────────────
void processCommand(const char* cmd) {
  if (strcmp(cmd, "PING") == 0) {
    Serial.println("OK");
    return;
  }

  if (cmd[0] == 'M' && cmd[1] == ',') {
    int left = 0, right = 0;
    int parsed = sscanf(cmd + 2, "%d,%d", &left, &right);
    if (parsed == 2) {
      // Clamp to safe range
      left  = constrain(left,  -MAX_SPEED, MAX_SPEED);
      right = constrain(right, -MAX_SPEED, MAX_SPEED);
      setMotors(left, right);
      lastCmdMs = millis();
      watchdogArmed = true;
      Serial.println("OK");
    } else {
      Serial.println("ERR,parse_failed");
    }
    return;
  }

  if (strcmp(cmd, "STOP") == 0) {
    stopMotors();
    watchdogArmed = false;
    Serial.println("OK");
    return;
  }

  Serial.println("ERR,unknown_command");
}

// ─── Motor Control ────────────────────────────────────────────────────────────
void setMotors(int left, int right) {
  setMotorLeft(left);
  setMotorRight(right);
}

void setMotorLeft(int speed) {
  if (speed > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else if (speed < 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    speed = -speed;
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
  }
  uint8_t pwm = map(speed, 0, MAX_SPEED, 0, 255);
  ledcWrite(PWM_CH_LEFT, pwm);
}

void setMotorRight(int speed) {
  if (speed > 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else if (speed < 0) {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    speed = -speed;
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
  }
  uint8_t pwm = map(speed, 0, MAX_SPEED, 0, 255);
  ledcWrite(PWM_CH_RIGHT, pwm);
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(PWM_CH_LEFT,  0);
  ledcWrite(PWM_CH_RIGHT, 0);
}
