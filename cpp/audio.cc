#include <iostream>
#include <fstream>
#include <valarray>
#include <vector>
#include <string>

// https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

using namespace std;

const float PI = 3.14159265358979323846;

class AudioPreparer {

    public:

        static std::valarray<float> load(std::string path) {

            // count length
            std::fstream a(path.c_str());
            std::string l;
            int signal_length = 0;
            while (std::getline(a, l)) ++signal_length;

            std::valarray<float> raw(signal_length);
            int idx = 0;

            std::fstream audiof(path.c_str());
            std::string line;
            while (std::getline(audiof, line)) {
                raw[idx] = std::stof(line);
                ++idx;
            }
            return raw;
        }

        static std::vector<std::valarray<float>> frame(std::valarray<float> wv, int win_length, int hop_length) {
            int offset = 0;
            std::vector<std::valarray<float>> frames;

            while (offset < wv.size() - win_length) {
                frames.push_back( wv[std::slice(offset, win_length, 1)] );
                offset += hop_length;
            }
            return frames;
        };

        static void hamming(std::valarray<float> window, int length) {
            for(int i = 0; i < window.size(); ++i) {
                window[i] = 0.54 - (0.46 * cos( (2 * PI * window[i]) / (length - 1) ));
            }
        };

        static void stft() {};

        static void filter_bank() {};

        static void write(std::vector<std::valarray<float>> frames, std::string path) { };

};

int main() {

    std::string wave_path = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/sample_wave.out";
    std::valarray<float> raw_wave = AudioPreparer::load(wave_path);
    int sr = 16000;
    int win_length = sr * 0.025;
    int hop_length = sr * 0.01; 

    // test framing
    std::vector<std::valarray<float>> frames = AudioPreparer::frame(raw_wave, win_length, hop_length);
    AudioPreparer::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/orig_frames.txt");

    // test Hamming window
    int window_length = 256;
    for (const auto& frame : frames) {
        AudioPreparer::hamming(frame, window_length);
    }
    AudioPreparer::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/hamming_frames.txt");

    // test stft
    for (const auto& frame : frames) {
        AudioPreparer::stft();
    }

    // test filter_bank
    for (const auto& frame : frames) {
        AudioPreparer::filter_bank();
    }

}