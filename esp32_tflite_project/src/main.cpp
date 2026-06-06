/*
 * ESP32 TensorFlow Lite Micro — CIC-IDS2017 IDS inference benchmark (78-dim input).
 *
 * Deploy model:
 *   python scripts/deploy_microcontroller.py \
 *     --model models/tflite/saved_model_pruned_qat.tflite \
 *     --output esp32_tflite_project/src/model_data.c \
 *     --array-name ids_tflite_model
 *
 * Serial output (for scripts/collect_esp32_benchmark.py):
 *   BENCHMARK latency_us=<us> arena_used=<bytes> input_dim=78
 */

#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

namespace {
constexpr int kInputDim = 78;
constexpr int kBenchmarkRuns = 20;
// IDS INT8 models typically need 80–120 KB arena on ESP32-S3; increase if AllocateTensors fails.
constexpr int kTensorArenaSize = 120 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Deterministic test vector (repeat pattern); replace with real preprocessed features if needed.
void fill_test_input(TfLiteTensor* tensor) {
  const int n = tensor->dims->size > 1 ? tensor->dims->data[1] : kInputDim;
  if (tensor->type == kTfLiteFloat32) {
    for (int i = 0; i < n; ++i) {
      tensor->data.f[i] = 0.01f * static_cast<float>((i % 10) + 1);
    }
  } else if (tensor->type == kTfLiteInt8) {
    for (int i = 0; i < n; ++i) {
      tensor->data.int8[i] = static_cast<int8_t>((i % 7) - 3);
    }
  }
}

void print_tensor_shape(TfLiteTensor* tensor, const char* name) {
  Serial.print(name);
  Serial.print(" shape: [");
  for (int i = 0; i < tensor->dims->size; i++) {
    Serial.print(tensor->dims->data[i]);
    if (i < tensor->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================");
  Serial.println("ESP32 IDS TFLite Micro Benchmark");
  Serial.println("========================================\n");

  if (ids_tflite_model_len <= 1) {
    Serial.println("ERROR: Placeholder model detected.");
    Serial.println("Run deploy_microcontroller.py to flash a real TFLite model.");
    return;
  }

  Serial.printf("Model size: %u bytes (%.2f KB)\n", ids_tflite_model_len,
                ids_tflite_model_len / 1024.0f);

  model = tflite::GetModel(ids_tflite_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Unsupported TFLite schema version");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("ERROR: AllocateTensors failed (arena=%d bytes)\n", kTensorArenaSize);
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  print_tensor_shape(input, "Input");
  print_tensor_shape(output, "Output");
  Serial.printf("Tensor arena size: %d bytes\n\n", kTensorArenaSize);

  // Warmup
  fill_test_input(input);
  interpreter->Invoke();

  for (int run = 0; run < kBenchmarkRuns; ++run) {
    fill_test_input(input);
    unsigned long t0 = micros();
    TfLiteStatus status = interpreter->Invoke();
    unsigned long t1 = micros();

    if (status != kTfLiteOk) {
      Serial.println("ERROR: Invoke failed");
      break;
    }

    float score = 0.0f;
    if (output->type == kTfLiteFloat32) {
      score = output->data.f[0];
    } else if (output->type == kTfLiteInt8) {
      score = output->data.int8[0] / 128.0f;
    }

    Serial.printf(
        "BENCHMARK latency_us=%lu arena_used=%d input_dim=%d score=%.4f run=%d\n",
        t1 - t0,
        interpreter->arena_used_bytes(),
        input->dims->size > 1 ? input->dims->data[1] : kInputDim,
        score,
        run + 1);
    delay(50);
  }

  Serial.println("\nBenchmark complete.");
  Serial.println("========================================\n");
}

void loop() {
  delay(10000);
}
