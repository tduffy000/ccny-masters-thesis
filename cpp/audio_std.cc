#include "iohandler.h"
#include <complex>
#include <valarray>
#include <vector>
#include <string>

/**
 * This file contains the Feature Engineering pipeline transforming
 * a raw normalized [-1, 1] audio amplitude waveform into a set of 
 * filter banks (Spectrogram) using the STL of C++.  
 */

using namespace std;

const float PI = 3.14159265358979323846;
const int SIGNAL_RATE = 16000;
const int WIN_LENGTH = SIGNAL_RATE * 0.025;
const int HOP_LENGTH = SIGNAL_RATE * 0.01; 
const int N_FFT = 512;

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

        static Waveform pad(Waveform& window, int offset) {
            Waveform padded(window.size() + 2 * offset);
            for (int i = 0; i < window.size(); i++) {
                padded[i+offset] = window[i];
            }
            return padded;
        }

        /** FOURIER TRANSFORM */
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

        /** FILTER BANKS */
        static std::valarray<float> mel_filters(int nfilter, int sr) {
            float low_freq = 0.0f;
            float high_freq = (2595 * std::log10(1 + (sr / 2)/ 700.0f));
            float step = (high_freq - low_freq) / (nfilter+1);

            std::valarray<float> filters (0.0f, nfilter+2);
            for (int i = 1; i < filters.size(); i++) {
                filters[i] = filters[i-1] + step;
            }
            return filters;
        };

        static std::valarray<float> mel_to_hz(std::valarray<float> filts) {
            return (700 * (std::pow(10, filts / 2595.0f) - 1));
        };

        static void magnitude_to_db(RealMatrix& a) {
            for(size_t row = 0; row < a.size(); row++) {
                a[row] = 20.0f * std::log10(a[row]);
            }
        };

        static void power_to_db(RealMatrix& a) {
            for(size_t row = 0; row < a.size(); row++) {
                a[row] = 10.0f * std::log10(a[row]);
            }
        };

        static RealMatrix transpose(RealMatrix& a) {
            size_t cols = a[0].size();
            size_t rows = a.size();
            RealMatrix transposed (cols);
            for (size_t i = 0; i < cols; i++ ) {
                transposed[i] = std::valarray<float> (rows);
                for (size_t j = 0; j < rows; j++) {
                    transposed[i][j] = a[j][i];
                }
            }
            return transposed;
        }

        // (n,k) x (k,m) => (n,m)
        static RealMatrix dot_product(RealMatrix& a, RealMatrix& b) {
            size_t a_rows = a.size();
            size_t a_cols = a[0].size();
            size_t b_rows = b.size();
            size_t b_cols = b[0].size();
            RealMatrix dot_prod (a_rows); // [a_rows x a_cols] * [b_rows x b_cols] |=> [a_rows x b_cols]

            // initialize just above zero for numerical stability in log
            for (size_t r = 0; r < a_rows; r++) dot_prod[r] = std::valarray<float> (0.000001f, b_cols);

            for (size_t i = 0; i < a_rows; ++i) {
                for (size_t j = 0; j < b_cols; ++j ) {
                    for (size_t k = 0; k < a_cols; ++k) {
                        dot_prod[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            return dot_prod;
        };

        static RealMatrix filter_banks(RealMatrix m, int nfilter = 40, int sr = 16000, int n_fft = 512) {
            std::valarray<float> filts = mel_filters(nfilter, sr);
            std::valarray<float> hz_points = mel_to_hz(filts);

            std::valarray<float> bins (hz_points.size());
            for (int i = 0; i < bins.size(); i++) {
                bins[i] = std::floor((n_fft + 1) * hz_points[i] / sr);
            }

            RealMatrix fb (nfilter);
            for (int m = 1; m < nfilter + 1; m++) {
                int f_m_minus = bins[m-1];
                int f_m = bins[m];
                int f_m_plus = bins[m+1];

                fb[m-1] = std::valarray<float> (n_fft / 2 + 1);
                for (int k = f_m_minus; k < f_m; k++) {
                    fb[m - 1][k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1]);
                }
                for (int k = f_m; k < f_m_plus; k++) {
                    fb[m - 1][k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m]);
                }
            }

            RealMatrix fb_T = transpose(fb);
            RealMatrix filter_banks = dot_product(m, fb_T);

            return filter_banks;
        };

        /** MFCCs */
        // static RealMatrix mfcc(RealMatrix m) {

        // };

};

int main() {

    std::string wave_path = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/sample_wave.out";
    AudioFeatures::Waveform raw_wave = IOHandler::load(wave_path);

    std::vector<AudioFeatures::Waveform> frames = AudioFeatures::frame(raw_wave, WIN_LENGTH, HOP_LENGTH);
    IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/std/orig_frames.txt");

    AudioFeatures::ComplexMatrix m = AudioFeatures::stft(frames, N_FFT);
    AudioFeatures::RealMatrix magnitude_mat = AudioFeatures::magnitude(m);
    IOHandler::write(magnitude_mat, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/std/stft_magnitude_frames.txt");

    AudioFeatures::RealMatrix power_mat = AudioFeatures::power(m);
    IOHandler::write(power_mat, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/std/stft_power_frames.txt");

    AudioFeatures::RealMatrix fb = AudioFeatures::filter_banks(magnitude_mat);
    AudioFeatures::magnitude_to_db(fb);
    IOHandler::write(AudioFeatures::transpose(fb), "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/std/filter_banks.txt");

    // // test mfcc
    // for (auto& frame : frames) {
    //     AudioPreparer::mfcc();
    // }
    // IOHandler::write(frames, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/mfcc.txt");
}