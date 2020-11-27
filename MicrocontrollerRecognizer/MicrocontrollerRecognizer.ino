#include <PDM.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "feature_provider.h"

// define EMBEDDING LEN when importing

namespace {

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
feature::FeatureProvider* feature_provider = nullptr;

TfLiteTensor* model_input = nullptr;

constexpr int tensorArenaSize = 10 * 1024;
uint8_t tensor_arena[tensorArenaSize];

const int signal_rate = 16000;
constexpr size_t waveform_length = 16000 * 1.2;       // SIGNAL_RATE * seconds
constexpr size_t window_length = signal_rate * 0.025; // SIGNAL_RATE * seconds
constexpr size_t hop_length = signal_rate * 0.01;     // SIGNAL_RATE * seconds
const int nfft = 512;
const int num_frames = 121;
const int n_filter = 40;
const size_t embedding_len = 128; 
const int noise_threshold = 20; // begin sampling if Audio exceeds this level

float raw_waveform_buffer[waveform_length];
float feature_buffer[n_filter][num_frames]; 
float embedding_buffer[embedding_len];

// we can enroll offline & import the embedding vector for our demo
bool is_buffer_full = false;

} // namespace

void setup() {

  // init the PDM mic
  PDM.onReceive(receivePDMData);
  PDM.setGain(20);
  constexpr int pdmBufferSize = waveform_length * 2;
  PDM.setBufferSize(pdmBufferSize);

  // setup the model stuff
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(speaker_model); // speaker_model defined in model.h

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

  static feature::FeatureProvider fp(waveform_length, raw_waveform_buffer, window_length, hop_length, n_filter, signal_rate, nfft, num_frames);
  feature_provider = &fp;

}

void loop() {

  if (!PDM.begin(1, signal_rate)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  // convert Mic input to normalized waveform [-1, 1]
  feature::normalize_waveform(raw_waveform_buffer, waveform_length);

  // turn into a spectrogram feature
  feature_provider->waveform_to_feature(feature_buffer);

  // copy? into model input space

  // call model to get embedding vector

  // print out cosine similarity with target embedding 
//  float similarity = MatrixMath::cosine_similarity(embedding_buffer, enrolled_embedding, embedding_len);

  // (optional) threshold to accept/reject
//  Serial.println(similarity);

};

void receivePDMData() {
// copy buffer into rawWaveformBuffer
//  int bytesAvailable = PDM.available();

//  Int bytesRead = PDM.read();
};
