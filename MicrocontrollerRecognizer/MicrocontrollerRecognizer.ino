#include <PDM.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "audio_feature_provider.h"

namespace {

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* model_input = nullptr;

constexpr int tensorArenaSize = 10 * 1024;
uint8_t tensor_arena[tensorArenaSize];

const int signal_rate = 16000;
constexpr size_t waveform_length = 16000 * 1.2;       // SIGNAL_RATE * seconds
constexpr size_t window_length = signal_rate * 0.025; // SIGNAL_RATE * seconds
constexpr size_t hop_length = signal_rate * 0.01;     // SIGNAL_RATE * seconds
const int n_fft = 512;
const int num_frames = 121;
const int n_filter = 40;
const size_t embedding_len = 128;
const int NOISE_THRESHOLD = 20; // begin sampling if Audio exceeds this level

} // namespace


// allocate the raw waveform blocks
float * raw_waveform = new float[waveform_length];

// allocate the spectrogram block
float ** feature = new float*[n_filter];

// allocate the embbeding target embedding vector
//float embedding_vector[embedding_len]; 

void setup() {

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(speaker_model);

  tflite::AllOpsResolver resolver;

  // will need to experiment to determine what this should be
  const int tensor_arena_size = 2 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);

  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  model_input = interpreter->input(0);

  PDM.onReceive(receivePDMData);

}

void loop() {

  if (!PDM.begin(1, signal_rate)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

   // convert Mic input to normalized waveform [-1, 1]

   // turn into a feature
   //  AudioFeatureProvider::waveform_to_feature();

   // pass through model to get embedding vector

   // print out cosine similarity with target embedding 
   //  float similarity = MatrixMath::cosine_similarity();

   // (optional) threshold to accept/reject
};

void receivePDMData() {
//  int bytesAvailable = PDM.available();

//  Int bytesRead = PDM.read();
};
