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

        // DRY this up, us the iterator template
        static void write(std::valarray<std::valarray<float>> mat, std::string path) {
            std::ofstream out_file;
            std::ostream_iterator<float> out_it (out_file, ",");

            out_file.open (path);
            for (const auto& row : mat) {
                std::copy(std::begin(row), std::end(row), out_it);
                out_file << std::endl;
            }
            out_file.close();
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
        typedef std::complex<float> Complex;
        typedef std::valarray<float> Waveform;
        typedef std::valarray<std::valarray<float>> Matrix;

        // https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html        
        // https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
        // https://cp-algorithms.com/algebra/fft.html
        // https://github.com/numpy/numpy/blob/92ebe1e9a6aeb47a881a1226b08218175776f9ea/numpy/fft/_pocketfft.py#L287
        static void fft(std::valarray<Complex>& x) {
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

        static std::vector<Waveform> frame(Waveform wv, int win_length, int hop_length) {
            int offset = 0;
            std::vector<Waveform> frames;

            while (offset < wv.size() - win_length) {
                frames.push_back( wv[std::slice(offset, win_length, 1)] );
                offset += hop_length;
            }
            return frames;
        };

        static void hamming(Waveform& window, int length) {
            for(int i = 0; i < window.size(); i++) {
                window[i] *= 0.54 - (0.46 * std::cos( (2 * PI * i) / (length - 1) ));
            }
        };

        // I'm not certain this is the implementation they use
        static Waveform pad(Waveform& window, int offset) {
            Waveform padded(window.size() + 2 * offset);
            for (int i = 0; i < window.size(); i++) {
                padded[i+offset] = window[i];
            }
            return padded;
        }

        // https://github.com/librosa/librosa/blob/a53fa56bdb6695a994008d5b6ccd0100870a6036/librosa/core/spectrum.py#L42
        static Matrix stft(std::vector<Waveform>& windows, int nfft = 512, float power = 1.0f) {
            
            bool to_pad = windows[0].size() < nfft;
            int length = to_pad ? nfft : windows[0].size();

            Matrix m (windows.size());

            int offset = to_pad ? (nfft - windows[0].size())/2 : 0;
            for (int i = 0; i < windows.size(); i++) {

                Waveform padded = pad(windows[i], offset);
                hamming(padded, nfft);
                std::valarray<Complex> complex_window (padded.size());
                for(int i = 0; i < padded.size(); i++) {
                    complex_window[i] = (Complex) padded[i];
                }
                fft(complex_window);

                std::valarray<float> real_part (complex_window.size());
                for (int i = 0; i < complex_window.size(); i++) {
                    real_part[i] = std::abs(complex_window[i]);
                }
                m[i] = real_part;
            }

            return m;
        };

        // https://github.com/librosa/librosa/blob/master/librosa/feature/spectral.py#L1873
        static void filter_bank() {};

        static void mfcc() {};

};

int main() {

    std::string wave_path = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/sample_wave.out";
    AudioPreparer::Waveform raw_wave = IOHandler::load(wave_path);

    // test framing
    std::vector<AudioPreparer::Waveform> frames = AudioPreparer::frame(raw_wave, WIN_LENGTH, HOP_LENGTH);
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/orig_frames.txt");

    // test Hamming window
    // for (auto& frame : frames) {
    //     AudioPreparer::hamming(frame, WIN_LENGTH);
    // }
    // IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/hamming_frames.txt");

    // test stft
    AudioPreparer::Matrix m = AudioPreparer::stft(frames, N_FFT);
    IOHandler::write(m, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/stft_frames.txt");

    // test filter_bank
    // for (auto& frame : frames) {
    //     AudioPreparer::filter_bank();
    // }
    // IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/filter_bank.txt");

    // // test mfcc
    // for (auto& frame : frames) {
    //     AudioPreparer::mfcc();
    // }
    // IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/mfcc.txt");
}