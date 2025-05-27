#include <MicroTFLite.h>

#include "model.h"
#include <Arduino_OV767X.h>

unsigned short pixels[176 * 144];  // QCIF: 176x144 (RGB565)
const int cropWidth = 96;
const int cropHeight = 96;

constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

#define LED_PIN LED_BUILTIN
#define LEDR 22  // 빨간 LED 핀
#define LEDG 23  // 초록 LED 핀

void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(LED_PIN, OUTPUT);
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);

  if (!Camera.begin(QQVGA, RGB565, 1)) {
    Serial.println("Camera initialization failed!");
    while (true);
  }

  Serial.println("Person Detection Inference Start");

  if (!ModelInit(model, tensor_arena, kTensorArenaSize)) {
    Serial.println("Model initialization failed!");
    while (true);
  }

  ModelPrintMetadata();
}

void loop() {
  Camera.readFrame(pixels);

  const int image_width = 176;
  const int image_height = 144;

  int offset_x = (image_width - cropWidth) / 2;
  int offset_y = (image_height - cropHeight) / 2;

  int input_index = 0;

  for (int y = 0; y < cropHeight; y++) {
    for (int x = 0; x < cropWidth; x++) {
      uint16_t pixel = pixels[(offset_y + y) * image_width + (offset_x + x)];

      // RGB565 바로 → Grayscale (정수 연산 기반)
      uint8_t r5 = (pixel >> 11) & 0x1F;
      uint8_t g6 = (pixel >> 5) & 0x3F;
      uint8_t b5 = pixel & 0x1F;

      // RGB → Grayscale (정수 가중치로 계산)
      uint32_t r = r5 * 299;   // ≈ 0.299 * 1000
      uint32_t g = g6 * 587;   // ≈ 0.587 * 1000
      uint32_t b = b5 * 114;   // ≈ 0.114 * 1000
      uint8_t gray = (r + g + b + 500) / 1000;

      // int8 정규화 후 float 변환
      int8_t qval = static_cast<int8_t>(gray - 128);
      ModelSetInput(static_cast<float>(qval) / 128.0f, input_index++);
    }
  }

  if (!ModelRunInference()) {
    Serial.println("Inference failed");
    return;
  }

  float score0 = ModelGetOutput(0);  // no person
  float score1 = ModelGetOutput(1);  // person

  float threshold = 0.5;  // 인식률 보정
  bool person_detected = (score0 > threshold);

  if (person_detected) {
    Serial.println(">> PERSON DETECTED <<");
    Serial.print("person : ");
    Serial.print(score0);
    Serial.print("              no person : ");
    Serial.println(score1);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
  } else {
    Serial.print("person : ");
    Serial.print(score0);
    Serial.print("              no person : ");
    Serial.println(score1);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
  }
}
