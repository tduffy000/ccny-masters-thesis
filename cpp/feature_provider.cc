#include <iostream>
#include "complex.h"
#include "matrix_math.h"

#define PI 3.1415926535897932384626433832795

#ifndef N_FILTER
#define N_FILTER 40
#endif

#ifndef N_FRAME
#define N_FRAME 121
#endif

// delete the things that were initialized with new
// https://blogs.sw.siemens.com/embedded-software/2014/05/27/problems-with-pointers-out-of-scope-out-of-mind/s
// https://stackoverflow.com/questions/14857246/when-a-pointer-is-created-in-scope-what-happens-to-the-pointed-to-variable-when

namespace feature {

static void normalize_waveform(float wv[], const int wave_length, int max_byte = 32768) {
    for (int i = 0; i < wave_length; i++) wv[i] /= max_byte;
}

enum FeatureStatus {
    ready = 0,
    error = 1
};

class FeatureProvider {

    private:
        uint win_length;
        uint hop_length;
        uint sr;
        uint n_fft;
        uint nfft_real; 
        uint n_filter;
        uint n_frame;

        // ref to source raw waveform
        size_t waveform_length;
        float* waveform_data;

        // blocks for intermediate computation
        float** frames;
        Complex** stft_frames;
        float** energies; 
        float** transposed;

        // static computation blocks
        float** fb;

        static void hamming(float window[], int window_size) {
            for(int i = 0; i < window_size; i++) {
                window[i] *= 0.54 - (0.46 * cos( (2 * PI * i) / (window_size - 1) ));
            }
        };

        /**
         * Perform the Short-term Fourier transform on each of the windows
         * which we framed above. Then take the magnitude / power of that transformation.
         */
        static void fft(Complex x[], int n) {
            if (n <= 1) return;

            int mid = n/2;
            Complex even [mid];
            Complex odd [mid];
            for (int i = 0; i < n; i++) {
                int idx = i / 2;
                if (i % 2 == 0) {
                even[idx] = x[i];
                } else {
                odd[idx] = x[i];
                }
            }

            fft(even, mid);
            fft(odd, mid);

            for (int k = 0; k < n/2; ++k) {
                Complex t = Complex::polar(1.0f, -2 * PI * k / n) * odd[k];
                x[k] = even[k] + t;
                x[k+n/2] = even[k] - t;
            }
        };

        /**
         * Frame the audio into overlapping windows, padding with zeros
         * to ensure each window is of length >= N_FFT.
         */
        void frame() {

            // pad the waveform on either side with nfft//2 with reflection
            uint wave_pad = n_fft / 2;
            float padded_waveform[waveform_length + n_fft];
            for (int l = 0; l < wave_pad; l++) {
                padded_waveform[wave_pad - l] = waveform_data[l];
            }
            for (int m = 0; m < waveform_length; m++) {
                padded_waveform[wave_pad + m] = waveform_data[m];
            }
            for (int r = 0; r < wave_pad; r++) {
                padded_waveform[wave_pad + waveform_length + r] = waveform_data[waveform_length - r - 1]; 
            }

            bool pad_frame = n_fft > win_length;
            uint frame_length = pad_frame ? n_fft : win_length;
            uint offset = pad_frame ? (n_fft - win_length) / 2 : 0;
            uint start = 0;

            for (int i = 0; i < n_frame; i++) {

                for (int j = 0; j < offset; j++) frames[i][j] = 0.0;
                for (int k = 0; k < win_length; k++) {
                    frames[i][offset+k] = padded_waveform[start+k];
                }
                for (int l = offset + win_length; l < frame_length; l++) frames[i][l] = 0.0;

                hamming(frames[i], n_fft);
                start += hop_length;
            }

        }

        static void stft(Complex** out, float** windows, uint num_frames, uint frame_length) {

            for (int i = 0; i < num_frames; i++) {
                
                Complex stft_frame[frame_length];
                for (int j = 0; j < frame_length; j++) {
                    stft_frame[j] = Complex (windows[i][j], 0.0f);
                }
                fft(stft_frame, frame_length);

                // take only the LHS; b/c real-valued signal means this is reflection symmetric
                for (int k = 0; k < frame_length / 2 + 1; k++) {
                    out[i][k] = stft_frame[k];
                }

              }

        };

        static void to_energy(float** out_frames, Complex** in_frames, uint num_frames, uint frame_length) {
            for (int i = 0; i < num_frames; i++) {
                for (int j = 0; j < frame_length; j++) {
                    out_frames[i][j] = in_frames[i][j].absolute_value();
                }
            }
        }

        static void mel_filters(float filters[], uint nfilter, uint sr) {
            float low_freq = 0.0;
            float high_freq = (2595 * std::log10(1 + (sr/2)/ 700.0f));
            float step = (high_freq - low_freq) / (nfilter+1);

            filters[0] = 0.0f;
            for (int i = 1; i < nfilter+2; i++) {
                filters[i] = filters[i-1] + step;
            }
        };

        static void mel_to_hz(float x[], int size) {
            for (int i = 0; i < size; i++ ) {
                x[i] = (700 * (std::pow(10, x[i] / 2595.0f) - 1));
            }
        };

        static void filter_bank(float** weights, uint n_mels, uint sr, uint nfft) {

            float filts[n_mels+2];
            mel_filters(filts, n_mels, sr);
            mel_to_hz(filts, n_mels+2);

            // difference between mel steps
            float fdiff[n_mels+1];
            for (int i = 0; i < n_mels + 1; i++) {
                fdiff[i] = filts[i+1] - filts[i];
            }

            // FFT frequencies
            float fft_freqs[1 + nfft / 2];
            float fft_freq_step = (sr * 1.0 / 2) / (nfft / 2);
            float fft_freq = 0.0;
            for (int i = 0; i < 1 + nfft / 2; i++) {
                fft_freqs[i] = fft_freq;
                fft_freq += fft_freq_step;
            }

            // outer subtraction: filts - fft_freqs
            float ramps[n_mels+2][1 + nfft/2];
            for (int i = 0; i < n_mels+2; i++) {
                for (int j = 0; j < 1 + nfft/2; j++) {
                    ramps[i][j] = filts[i] - fft_freqs[j]; 
                }
            }

            for (int i = 0; i < n_mels; i++) {

                for (int j = 0; j < 1 + nfft/2; j++) {
                    float lower = -1.0 * ramps[i][j] / fdiff[i];
                    float upper = ramps[i+2][j] / fdiff[i+1];
                    float bound = lower < upper ? lower : upper;
                    weights[i][j] = 0.0 > bound ? 0.0 : bound;
                }
            }

            // Slaney normalize
            float enorm;
            for (int i = 0; i < n_mels; i++) {
                enorm = 2.0 / (filts[i+2] - filts[i]);
                for (int j = 0; j < 1 + nfft/2; j++) {
                    weights[i][j] *= enorm;
                }
            }
        }

        template<size_t rows, size_t cols>
        static void log_magnitude(float (&feature_buffer)[rows][cols]) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    feature_buffer[i][j] = std::log10(std::pow(feature_buffer[i][j], 2.0) + 0.000001);
                }
            }
        };

        static void log_magnitude(float (&feature_buffer)[N_FILTER][N_FRAME]) {
            for (int i = 0; i < N_FILTER; i++) {
                for (int j = 0; j < N_FRAME; j++) {
                    feature_buffer[i][j] = std::log10(std::pow(feature_buffer[i][j], 2.0) + 0.000001);
                }
            }
        };

        static void log_magnitude(float** mat, uint n_rows, uint n_cols) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    mat[i][j] = std::log10(std::pow(mat[i][j], 2.0) + 0.000001);
                }
            }
        };

    public:

        FeatureProvider(
            size_t waveform_length,
            float* waveform_data,
            uint win_length,
            uint hop_length,
            uint n_filter,
            uint sr, 
            uint n_fft,
            uint n_frame
        )   : waveform_length(waveform_length),
            waveform_data(waveform_data),
            sr(sr),
            n_filter(n_filter),
            n_fft(n_fft),
            nfft_real(n_fft / 2 + 1),
            win_length(win_length),
            hop_length(hop_length),
            n_frame(n_frame) {

            // allocate the block for raw waveform data
            for (size_t i = 0; i < waveform_length; i++) waveform_data[i] = 0;

            // allocate the re-usable filter bank block
            fb = new float*[n_filter]; // [n_filter, nfft / 2 + 1]
            for (uint i = 0; i < n_filter; i++) fb[i] = new float[nfft_real]; 
            filter_bank(fb, n_filter, sr, n_fft);

            // allocate all the temporary computation blocks
            frames = new float*[n_frame]; // [n_frame, nfft]
            for (uint i = 0; i < n_frame; i++) frames[i] = new float[n_fft]; 

            stft_frames = new Complex*[n_frame]; // [n_frame, nfft / 2 + 1]
            for (uint i = 0; i < n_frame; i++) stft_frames[i] = new Complex[nfft_real];
            
            energies = new float*[n_frame]; // [n_frame, nfft / 2 + 1]
            for (uint i = 0; i < n_frame; i++) energies[i] = new float[nfft_real];
            
            transposed = new float*[nfft_real]; // [nfft / 2 + 1, n_frame]
            for (uint i = 0; i < nfft_real; i++) transposed[i] = new float[n_frame];
        };

        // TODO: this needs to delete EVERYTHING!!!!
        ~FeatureProvider() {};

        template<size_t rows, size_t cols>
        FeatureStatus waveform_to_feature(float (&feature_buffer)[rows][cols]) {
            
            uint nfft_real = n_fft / 2 + 1;
            frame();
            stft(stft_frames, frames, n_frame, n_fft);
            to_energy(energies, stft_frames, n_frame, nfft_real);

            MatrixMath::transpose(transposed, energies, n_frame, nfft_real);
            MatrixMath::dot_product<rows, cols>(feature_buffer, fb, transposed, n_filter, nfft_real, nfft_real, n_frame);

            log_magnitude<rows, cols>(feature_buffer);

            return FeatureStatus(0);
        };

        FeatureStatus waveform_to_feature(float (&feature_buffer)[N_FILTER][N_FRAME]) {
            
            uint nfft_real = n_fft / 2 + 1;
            frame();
            stft(stft_frames, frames, n_frame, n_fft);
            to_energy(energies, stft_frames, n_frame, nfft_real);

            MatrixMath::transpose(transposed, energies, n_frame, nfft_real);
            MatrixMath::dot_product<rows, cols>(feature_buffer, fb, transposed, n_filter, nfft_real, nfft_real, n_frame);

            log_magnitude<rows, cols>(feature_buffer);

            return FeatureStatus(0);
        };



};

} // namespace
