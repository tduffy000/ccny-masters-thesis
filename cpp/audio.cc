#include <iostream>
#include <iterator>
#include <complex>
#include <fstream>
#include <valarray>
#include <vector>
#include <string>

// https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

using namespace std;

const float PI = 3.14159265358979323846;
const int SIGNAL_RATE = 16000;
const int WIN_LENGTH = SIGNAL_RATE * 0.025;
const int HOP_LENGTH = SIGNAL_RATE * 0.01; 
const int N_FFT = 512;

class IOHandler {
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
        };

        static void write(std::vector<std::valarray<float>> frames, std::string path) {
            std::ofstream out_file;
            std::ostream_iterator<float> out_it (out_file, ",");

            out_file.open (path);
            for (const auto& f : frames) {
                std::copy(std::begin(f), std::end(f), out_it);
                out_file << std::endl;
            }
            out_file.close();
        };

};

class AudioPreparer {

    public:
        // https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html        
        // https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
        // https://cp-algorithms.com/algebra/fft.html
        typedef std::complex<float> Complex;

        static void fft(std::valarray<Complex> x) {
            const size_t N = x.size();
            if (N <= 1) return;

            std::valarray<Complex> even = x[std::slice(0, N/2, 2)];
            std::valarray<Complex> odd = x[std::slice(1, N/2, 2)];
        
            fft(even);
            fft(odd);

            for(size_t k = 0; k < N/2; ++k) {
                Complex t = std::polar(1.0f, -2 * PI * k / N) * odd[k];
                x[k] = even[k] + t;
                x[k+N/2] = even[k] - t;
            };
        };

        static std::vector<std::valarray<float>> frame(std::valarray<float> wv, int win_length, int hop_length) {
            int offset = 0;
            std::vector<std::valarray<float>> frames;

            while (offset < wv.size() - win_length) {
                frames.push_back( wv[std::slice(offset, win_length, 1)] );
                offset += hop_length;
            }
            return frames;
        };

        static void hamming(std::valarray<float>& window, int length) {
            for(int i = 0; i < window.size(); i++) {
                window[i] *= 0.54 - (0.46 * std::cos( (2 * PI * i) / (length - 1) ));
            }
        };

        // https://github.com/librosa/librosa/blob/a53fa56bdb6695a994008d5b6ccd0100870a6036/librosa/core/spectrum.py#L42
        static std::valarray<float> stft(std::valarray<float>& window, int nfft) {
            // if nfft > window.size(); do we pad with zeros?
            fft(window);
            return std::pow(std::abs(window), 2.0f) / nfft;
        };

        // https://github.com/librosa/librosa/blob/master/librosa/feature/spectral.py#L1873
        static void filter_bank() {};

        static void mfcc() {};

};

int main() {

    std::string wave_path = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/sample_wave.out";
    std::valarray<float> raw_wave = IOHandler::load(wave_path);

    // test framing
    std::vector<std::valarray<float>> frames = AudioPreparer::frame(raw_wave, WIN_LENGTH, HOP_LENGTH);
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/orig_frames.txt");

    // test Hamming window
    for (auto& frame : frames) {
        AudioPreparer::hamming(frame, WIN_LENGTH);
    }
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/hamming_frames.txt");

    // test stft
    for (auto& frame : frames) {
        AudioPreparer::stft(frame, N_FFT);
    }
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/stft_frames.txt");

    // test filter_bank
    for (auto& frame : frames) {
        AudioPreparer::filter_bank();
    }
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/filter_bank.txt");

    // test mfcc
    for (auto& frame : frames) {
        AudioPreparer::mfcc();
    }
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/mfcc.txt");
}