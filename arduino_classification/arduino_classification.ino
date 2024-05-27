// Includes
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "converted_test_images/image_0.h"
#include "model_pruned_quantized_10x.h"

// Constants and Global Variables
const float accelerationThreshold = 2.5;
const int numSamples = 119;
int samplesRead = numSamples;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
constexpr int tensorArenaSize = 24 * 1024;
alignas(16) byte tensorArena[tensorArenaSize];
const char* GESTURES[] = {"punch", "flex"};
const int NUM_GESTURES = sizeof(GESTURES) / sizeof(GESTURES[0]);

// Setup Function
void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println();

  // Load the model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (true);  // Halt execution
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true);  // Halt execution
  }

  // Get pointers to the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

// Loop Function
void loop() {
  // Normalize and set input data
  for (int i = 0; i < 28 * 28; ++i) {
    tflInputTensor->data.int8[i] = static_cast<uint8_t>(image_data[i] / 255.0f * 255);
  }

  // Run inference
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Process output
  float scale = tflOutputTensor->params.scale;
  int zero_point = tflOutputTensor->params.zero_point;
  uint8_t* probabilities = tflOutputTensor->data.uint8;

  float max_probability = -1.0;
  int max_index = -1;

  for (int i = 0; i < 10; ++i) {
    float probability = (probabilities[i] - zero_point) * scale;
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(probability, 6);

    if (probability > max_probability) {
      max_probability = probability;
      max_index = i;
    }
  }

  Serial.print("Prediction = ");
  Serial.print(max_index);
  Serial.print(", ");
  Serial.print("Probability = ");
  Serial.println(max_probability, 6);

  Serial.println();
  delay(10000);
}
