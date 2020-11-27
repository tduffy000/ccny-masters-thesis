#include "feature_provider.cc"
#include "iohandler.h"

#include <unistd.h>
#include <string>
#include <cstdlib>
#include <ctime>

#define N_FFT 512
#define N_FILTER 40
#define SR 16000
#define N_FRAME 121

namespace {

constexpr int waveform_length = SR * 1.2;
constexpr int win_length = SR * 0.025;
constexpr int hop_length = SR * 0.01;

const uint sleep_time = 1;

float raw_waveform_buffer[waveform_length]; 
float feature_buffer[N_FILTER][N_FRAME]; 

feature::FeatureProvider* feature_provider = nullptr;

}

void populate_random_waveform(float wv[], const int length) {
    for (int i = 0; i < length; i++) {
        float sign = (rand() > RAND_MAX/2) ? -1.0 : 1.0;
        wv[i] = (float(rand()) / float((RAND_MAX)) * sign);
    }
}

void setup() {
    static feature::FeatureProvider fp(waveform_length, raw_waveform_buffer, win_length, hop_length,
                                        N_FILTER, SR, N_FFT, N_FRAME);
    feature_provider = &fp;
};

void loop() {

    const std::string path_prefix = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/bare"; 
    std::string wv_path;
    std::string feature_path;
    uint idx = 0;

    while (true) {

        populate_random_waveform(raw_waveform_buffer, waveform_length);
        feature_provider->waveform_to_feature<N_FILTER, N_FRAME>(feature_buffer);
        
        wv_path = path_prefix + "/raw_wave_" + std::to_string(idx);
        feature_path = path_prefix + "/feature_" + std::to_string(idx);
        
        std::cout << "writing out raw waveform to: " << wv_path << std::endl;
        std::cout << "writing feature to: " << feature_path << std::endl;
        
        IOHandler::write(raw_waveform_buffer, waveform_length, wv_path);
        IOHandler::write<N_FILTER, N_FRAME>(feature_buffer, feature_path);

        sleep(sleep_time);
        ++idx;

    }
};

int main() {
    srand((unsigned int)time(0));
    setup();
    loop();
}