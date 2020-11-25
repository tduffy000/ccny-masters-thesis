#include "complex.h"
#include "matrix_math.h"

#define N_FFT 512
#define N_FILTER 40
#define SR 16000
#define N_FRAME 121

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
        int win_length;
        int hop_length;

        size_t waveform_size;
        float waveform_data[];
        float* filters;
        float fb[][N_FFT/2 + 1];

        // TODO: re-order these methods into logical blocks
        static void hamming(float window[], int window_size) {
            for(int i = 0; i < window_size; i++) {
                window[i] *= 0.54 - (0.46 * cos( (2 * PI * i) / (window_size - 1) ));
            }
        };

        /**
         * Frame the audio into overlapping windows, padding with zeros
         * to ensure each window is of length >= N_FFT.
         */
        void frame(float frames[N_FRAME][N_FFT], const int num_frames, float waveform[], int waveform_length, int win_length, const int hop_length, const int nfft) {

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

        static void stft(Complex out[N_FRAME][N_FFT / 2 + 1], float windows[N_FRAME][N_FFT], int num_frames, int frame_length) {

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

        static void to_energy(float out[N_FRAME][N_FFT / 2 + 1], Complex stft_frames[N_FRAME][N_FFT / 2 + 1], int num_frames, int frame_length) {
            for (int i = 0; i < num_frames; i++) {
                float* frame = new float[frame_length];
                for (int j = 0; j < frame_length; j++) {
                    frame[j] = stft_frames[i][j].absolute_value();
                }
                out[i] = frame;
            }
        }

        static void mel_filters(float filters[], int nfilter, int sr) {
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

        static void filter_bank(float weights[][N_FFT/2 + 1], int n_mels, int sr, int nfft) {

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

                float * w = new float[1+nfft/2];
                for (int j = 0; j < 1 + nfft/2; j++) {
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
                for (int j = 0; j < 1 + nfft/2; j++) {
                    weights[i][j] *= enorm;
                }
            }
        }

        static void log_magnitude(float mat[N_FILTER][N_FRAME], int n_rows, int n_cols) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    mat[i][j] = std::log10(std::pow(mat[i][j], 2.0) + 0.000001);
                }
            }
        };

    public:

        FeatureProvider(
            size_t waveform_size,
            float waveform_data[],
            int win_length,
            int hop_length
        )   : waveform_size(waveform_size),
            waveform_data(waveform_data),
            win_length(win_length),
            hop_length(hop_length) {

            // allocate the block for raw waveform data
            for (size_t i = 0; i < waveform_size; i++) waveform_data[i] = 0;

            // allocate the mel filter banks
            filter_bank(fb, N_FILTER, SR, N_FFT);
        };

        // does this need to delete the filters & spectrogram blocks?
        ~FeatureProvider() {};

        // what does this return? just a status that the caller can access it?
        // does the caller provide a place for us to dump the feature into?
        // do we want to de-allocate as we go with the intermediate steps?
        FeatureStatus waveform_to_feature(float feature_buffer[N_FILTER][N_FRAME]) {
            // frame
            float frames[N_FRAME][N_FFT];
            frame(frames, N_FRAME, waveform_data, waveform_size, win_length, hop_length, N_FFT);

            // stft + spectrum transformation
            Complex stft_frames[N_FRAME][N_FFT / 2 + 1];
            stft(stft_frames, frames, N_FRAME, N_FFT);

            // convert back to abs value
            float energy_frames[N_FRAME][N_FFT / 2 + 1];
            to_energy(energy_frames, stft_frames, N_FRAME, N_FFT / 2 + 1);

            // dot product
            MatrixMath::dot_product(feature_buffer, MatrixMath::transpose(spec_frames, N_FRAME, 1 + N_FFT/2), 40, 1+N_FFT/2, 1+N_FFT/2, N_FRAME);

            // log_magnitude
            log_magnitude(feature_buffer, N_FILTER, N_FRAME);

            return FeatureStatus(0);
        };


};

} // namespace