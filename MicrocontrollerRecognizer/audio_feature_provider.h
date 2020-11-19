#include "complex.h"

class MatrixMath {

    static void transpose(float ** out, float ** a, int rows, int cols) {
        for (int i = 0; i < cols; i++) {
            float* t_row = new float[rows];
            for (int j = 0; j < rows; j++) {
                t_row[j] = a[j][i];
            }
            out[i] = t_row;
        }
        // delete a
    }

    static void dot_product(float** out, float ** a, float ** b, int a_rows, int a_cols, int b_rows, int b_cols) {
        for (int r = 0; r < a_rows; r++) {
            float* row = new float[b_cols];
            for (int l = 0; l < b_cols; l++) row[l] = 0.0;
            out[r] = row;
        }

        for (int i = 0; i < a_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                for (int k = 0; k < a_cols; ++k) {
                    out[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        // delete a + b
    }

    float dot_product(float a[], float b[], size_t length) {
        float dot_product = 0.0;
        for (size_t i = 0; i < length; i++) dot_product += a[i] * b[i];
        return dot_product;
    };

    static void mean_normalize(float ** mat, int n_rows, int n_cols) {
        for (int i = 0; i < n_rows; i++) {
            float total = 0.0;
            for (int j = 0; j < n_cols; j++) {
                total += mat[i][j];
            }
            float mean = total / n_cols;
            for (int j = 0; j < n_cols; j++) {
                mat[i][j] -= mean;
            }
        }
    };

    float cosine_similarity(float a[], float b[], size_t length) {
        float numerator = dot_product(a, b, length);
        float a_norm = std::sqrt(dot_product(a, a, length));
        float b_norm = std::sqrt(dot_product(b, b, length));
        return numerator / (a_norm * b_norm);
    }

};

class AudioFeatureProvider {

    static void hamming(float window[], int window_size) {
        for(int i = 0; i < window_size; i++) {
            window[i] *= 0.54 - (0.46 * cos( (2 * PI * i) / (window_size - 1) ));
        }
    };

    /**
     * Frame the audio into overlapping windows, padding with zeros
     * to ensure each window is of length >= N_FFT.
     */
    static void frame(float** frames, const int num_frames, float waveform[], int waveform_length, int win_length, const int hop_length, const int nfft) {

        // pad the waveform on either side with nfft//2 with reflection
        int wave_pad = nfft / 2;
        float padded_waveform[waveform_length + nfft];
        for (int l = 0; l < wave_pad; l++) {
            padded_waveform[wave_pad - l] = waveform[l];
        }
        for (int m = 0; m < waveform_length; m++) {
            padded_waveform[wave_pad + m] = waveform[m];
        }
        for (int r = 0; r < wave_pad; r++) {
            padded_waveform[wave_pad + waveform_length + r] = waveform[waveform_length - r - 1]; 
        }

        bool pad_frame = nfft > win_length;
        int frame_length = pad_frame ? nfft : win_length;
        int offset = pad_frame ? (nfft - win_length) / 2 : 0;
        int start = 0;

        for (int i = 0; i < num_frames; i++) {

            float* frame = new float[frame_length];

            for (int j = 0; j < offset; j++) frame[j] = 0.0;
            for (int k = 0; k < win_length; k++) {
            frame[offset+k] = padded_waveform[start+k];
            }
            for (int l = offset + win_length; l < frame_length; l++) frame[l] = 0.0;

            hamming(frame, nfft);
            frames[i] = frame;
            start += hop_length;
        }

    }

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

    static void stft(Complex** out, float ** windows, int num_frames, int frame_length) {

        for (int i = 0; i < num_frames; i++) {
            
            Complex stft_frame[frame_length];
            for (int j = 0; j < frame_length; j++) {
                stft_frame[j] = Complex (windows[i][j], 0.0f);
            }
            fft(stft_frame, frame_length);

            // take only the LHS; b/c real-valued signal means this is reflection symmetric
            Complex* left_frame = new Complex[frame_length / 2 + 1];
            for (int k = 0; k < frame_length / 2 + 1; k++) {
                left_frame[k] = stft_frame[k];
            }

            out[i] = left_frame;
        }

    };

    static void to_energy(float** out, Complex ** stft_frames, int num_frames, int frame_length) {
        for (int i = 0; i < num_frames; i++) {
            float* frame = new float[frame_length];
            for (int j = 0; j < frame_length; j++) {
                frame[j] = stft_frames[i][j].absolute_value();
            }
            out[i] = frame;
        }
    }

    static void mel_filters(float* filters, int nfilter, int sr) {
        constexpr float low_freq = 0.0;
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

    static void filter_bank(float** weights, int n_mels, int sr, int n_fft) {

        float filts[n_mels+2];
        mel_filters(filts, n_mels, sr);
        mel_to_hz(filts, n_mels+2);

        // difference between mel steps
        float fdiff[n_mels+1];
        for (int i = 0; i < n_mels + 1; i++) {
            fdiff[i] = filts[i+1] - filts[i];
        }

        // FFT frequencies
        float fft_freqs[1 + n_fft / 2];
        float fft_freq_step = (sr * 1.0 / 2) / (n_fft / 2);
        float fft_freq = 0.0;
        for (int i = 0; i < 1 + n_fft / 2; i++) {
            fft_freqs[i] = fft_freq;
            fft_freq += fft_freq_step;
        }

        // outer subtraction: filts - fft_freqs
        float ramps[n_mels+2][1 + n_fft/2];
        for (int i = 0; i < n_mels+2; i++) {
            for (int j = 0; j < 1 + n_fft/2; j++) {
                ramps[i][j] = filts[i] - fft_freqs[j]; 
            }
        }

        for (int i = 0; i < n_mels; i++) {

            float * w = new float[1+n_fft/2];
            for (int j = 0; j < 1 + n_fft/2; j++) {
                float lower = -1.0 * ramps[i][j] / fdiff[i];
                float upper = ramps[i+2][j] / fdiff[i+1];
                float bound = lower < upper ? lower : upper;
                w[j] = 0.0 > bound ? 0.0 : bound;
            }
            weights[i] = w;
        }

        // Slaney normalize
        float enorm;
        for (int i = 0; i < n_mels; i++) {
            enorm = 2.0 / (filts[i+2] - filts[i]);
            for (int j = 0; j < 1 + n_fft/2; j++) {
                weights[i][j] *= enorm;
            }
        }
    }

    static void log_magnitude(float ** mat, int n_rows, int n_cols) {
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                mat[i][j] = std::log10(std::pow(mat[i][j], 2.0) + 0.000001);
            }
        }
    };

    static void waveform_to_feature(float waveform[], float** feature) {
        // build the spectrogram from the raw waveform
    };

};
