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

        static void write(std::valarray<std::valarray<float>> frames, std::string path) {
            std::ofstream out_file;
            std::ostream_iterator<float> out_it (out_file, ",");

            out_file.open (path);
            for (const auto& f : frames) {
                std::copy(std::begin(f), std::end(f), out_it);
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

class AudioFeatures {

    public:
        typedef std::complex<float> Complex;
        typedef std::valarray<float> Waveform;
        typedef std::valarray<std::valarray<Complex>> ComplexMatrix;
        typedef std::valarray<std::valarray<float>> RealMatrix;

        /** WINDOWING */
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

        /** Fourier transform */
        static RealMatrix raise(ComplexMatrix& mat, float power) {
            RealMatrix m (mat.size());
            for (int i = 0; i < mat.size(); i++) {
                std::valarray<float> x (mat[i].size());
                for (int j = 0; j < mat[i].size(); j++) {
                    x[j] = std::pow(std::abs(mat[i][j]), power);
                }
                m[i] = x;
            }
            return m;       
        }

        static RealMatrix magnitude(ComplexMatrix mat) {
            return raise(mat, 1.0f);
        };

        static RealMatrix power(ComplexMatrix mat) {
            return raise(mat, 2.0f);
        };

        // http://www.robots.ox.ac.uk/~sjrob/Teaching/SP/l7.pdf
        // https://www.princeton.edu/~cuff/ele201/files/spectrogram.pdf
        static std::valarray<Complex> dft(std::valarray<Complex> &in) {
            Complex J(0, 1);
            const size_t N = in.size(); 
            std::valarray<Complex> out (N);

            for (size_t k = 0; k < N; k++) {
                Complex s(0, 0);
                for (size_t t = 0; t < N; t++) {
                    float angle = 2 * PI * t * k / N;
                    s += in[t] * std::exp(Complex(0, -angle));
                }
                out[k] = s;
            }
            return out;
        };

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

        // https://github.com/librosa/librosa/blob/a53fa56bdb6695a994008d5b6ccd0100870a6036/librosa/core/spectrum.py#L42
        static ComplexMatrix stft(std::vector<Waveform>& windows, int nfft = 512) {
            
            bool to_pad = windows[0].size() < nfft;
            int length = to_pad ? nfft : windows[0].size();

            ComplexMatrix m (windows.size());

            int offset = to_pad ? (nfft - windows[0].size())/2 : 0;
            for (int i = 0; i < windows.size(); i++) {

                Waveform padded = pad(windows[i], offset);
                hamming(padded, nfft);

                std::valarray<Complex> complex_window (padded.size());
                for(int j = 0; j < padded.size(); j++) {
                    complex_window[j] = (Complex) padded[j];
                }
                fft(complex_window);
                // only need the left side because for real-valued signal, transform is reflection-symmetric
                std::valarray<Complex> real_side = complex_window[std::slice(0, 1 + (nfft/2), 1)];
                m[i] = real_side;
            }
            return m;
        };

        /** Filter banks */
        static std::valarray<float> mel_filters(int nfilter, int sr) {
            float low_freq = 0.0f;
            float high_freq = (2595 * std::log10(1 + (sr / 2)/ 700));
            float step = (high_freq - low_freq) / nfilter;

            std::valarray<float> filters (0.0f, nfilter+2);
            for (int i = 1; i < filters.size(); i++) {
                filters[i] = filters[i-1] + step;
            }
            return filters;
        };

        static std::valarray<float> mel_to_hz(std::valarray<float> filts) {
            return (700 * (std::pow(filts / 2595.0f, 10) - 1));
        }

        // https://github.com/librosa/librosa/blob/master/librosa/feature/spectral.py#L1873
        // static RealMatrix filter_bank(RealMatrix m) {

        // };

        /** MFCCs */
        // static RealMatrix mfcc(RealMatrix m) {

        // };

};

int main() {

    std::string wave_path = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/sample_wave.out";
    AudioFeatures::Waveform raw_wave = IOHandler::load(wave_path);

    // test framing
    std::vector<AudioFeatures::Waveform> frames = AudioFeatures::frame(raw_wave, WIN_LENGTH, HOP_LENGTH);
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/orig_frames.txt");

    // test stft
    AudioFeatures::ComplexMatrix m = AudioFeatures::stft(frames, N_FFT);
    AudioFeatures::RealMatrix magnitude_mat = AudioFeatures::magnitude(m);
    IOHandler::write(magnitude_mat, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/stft_magnitude_frames.txt");

    AudioFeatures::RealMatrix power_mat = AudioFeatures::power(m);
    IOHandler::write(power_mat, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/stft_power_frames.txt");

    // test filter_bank
    // AudioPreparer::RealMatrix fb = AudioPreparer::filter_bank(mat);
    // IOHandler::write(fb, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/filter_bank.txt");

    // // test mfcc
    // for (auto& frame : frames) {
    //     AudioPreparer::mfcc();
    // }
    // IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/mfcc.txt");
}