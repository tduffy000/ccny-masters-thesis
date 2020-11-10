#include "iohandler.h"
#include <iostream>

const float PI = 3.14159265358979323846;

/**
 * This file contains the Feature Engineering pipeline transforming
 * a raw normalized [-1, 1] audio amplitude waveform into a set of 
 * filter banks (Spectrogram) by hand given the library constraints
 * of an Arduino.
 * These are the same functions used in MicrocontrollerRecognizer.  
 */

class Complex {
  
  public:
    Complex(const float r = 0, const float i = 0) : re(r), im(i) {};

    float real () {return re;};
    float img () {return im;};

    float absolute_value() {
      return std::sqrt(std::pow(re, 2) + std::pow(im, 2));
    }

    Complex operator * (const Complex& x) {
      float r = re * x.re - im * x.im;
      float i = re * x.im + im * x.re;
      return Complex(r, i);
    }

    Complex operator - (const Complex& x) {
      return Complex(re - x.re, im - x.im);
    }

    Complex operator + (const Complex& x) {
      return Complex(re + x.re, im + x.im); 
    }
  
    static Complex polar(const float &rho, const float &theta) {
       return Complex(rho * cos(theta), rho * sin(theta));
    }
  protected:
    float re;
    float im;
};

/**
 * Feature Engineering
 */
const int SIGNAL_RATE = 16000;
const int WIN_LENGTH = SIGNAL_RATE * 0.025; // SIGNAL_RATE * seconds
const int HOP_LENGTH = SIGNAL_RATE * 0.01;  // SIGNAL_RATE * seconds
const int N_FFT = 512;
const int WAVEFORM_LENGTH = 16000 * 1.2;    // SIGNAL_RATE * seconds
const int NUM_FRAMES = 121;
const int N_FILTER = 40;

/**
 * Frame the audio into overlapping windows, padding with zeros
 * to ensure each window is of length >= N_FFT.
 */
void hamming(float window[], int window_size) {
    for(int i = 0; i < window_size; i++) {
        window[i] *= 0.54 - (0.46 * cos( (2 * PI * i) / (window_size - 1) ));
    }
};

float ** frame(float waveform[], int waveform_length, int win_length, const int hop_length, const int nfft) {

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

  float** frames = new float*[NUM_FRAMES];

  bool pad_frame = nfft > win_length;
  int frame_length = pad_frame ? nfft : win_length;
  int offset = pad_frame ? (nfft - win_length) / 2 : 0;
  int start = 0;

  for (int i = 0; i < NUM_FRAMES; i++) {

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

  return frames;
}

/**
 * Perform the Short-term Fourier transform on each of the windows
 * which we framed above. Then take the magnitude / power of that transformation.
 */
void fft(Complex x[], int n) {
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

Complex ** stft(float ** windows, int num_frames = NUM_FRAMES, int frame_length = N_FFT) {

  Complex** stft_frames = new Complex*[num_frames];

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

    stft_frames[i] = left_frame;
  }

  return stft_frames;
};

float ** to_energy(Complex ** stft_frames, int num_frames, int frame_length) {
  float** frames = new float*[num_frames];
  for (int i = 0; i < num_frames; i++) {
    float* frame = new float[frame_length];
    for (int j = 0; j < frame_length; j++) {
      frame[j] = stft_frames[i][j].absolute_value();
    }
    frames[i] = frame;
  }
  return frames; 
}

/**
 * Convert the power / magnitude spectrum into Filter banks (spectrogram), which
 * are the input features to our model.
 */
float * mel_filters(int nfilter, int sr = SIGNAL_RATE) {
  float low_freq = 0.0;
  float high_freq = (2595 * std::log10(1 + (sr/2)/ 700.0f));
  float step = (high_freq - low_freq) / (nfilter+1);

  float * filters = new float[nfilter+2];
  filters[0] = 0.0f;
  for (int i = 1; i < nfilter+2; i++) {
      filters[i] = filters[i-1] + step;
  }
  return filters;
}

void mel_to_hz(float x[], int size) {
  for (int i = 0; i < size; i++ ) {
    x[i] = (700 * (std::pow(10, x[i] / 2595.0f) - 1));
  }
};

float ** transpose(float ** a, int rows, int cols) {
  float ** t = new float*[cols];
  for (int i = 0; i < cols; i++) {
    float * t_row = new float[rows];
    for (int j = 0; j < rows; j++) {
      t_row[j] = a[j][i];
    }
    t[i] = t_row;
  }

  return t;
}

float ** dot_product(float ** a, float ** b, int a_rows, int a_cols, int b_rows, int b_cols) {

  float ** dot_prod = new float*[a_rows];

  for (int r = 0; r < a_rows; r++) {
    float* row = new float[b_cols];
    for (int l = 0; l < b_cols; l++) row[l] = 0.0;
    dot_prod[r] = row;
  }

  for (int i = 0; i < a_rows; ++i) {
    for (int j = 0; j < b_cols; ++j) {
      for (int k = 0; k < a_cols; ++k) {
        dot_prod[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return dot_prod;
}

float ** filter_bank(int n_mels, int sr = 16000, int n_fft = 512) {

  // mel scale
  float * filts = mel_filters(n_mels, sr);
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

  // TODO: everything above is const and, therefore, can be 
  // a parameter
  // now build our filter bank matrix
  float ** weights = new float*[n_mels];

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

  return weights;
}

void log_magnitude(float ** mat, int n_rows, int n_cols) {
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      mat[i][j] = std::log10(std::pow(mat[i][j], 2.0) + 0.000001);
    }
  }
};

void mean_normalize(float ** mat, int n_rows, int n_cols) {
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
}

/**
 * Test main()
 */
int main() {

    std::string wave_path = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/sample_wave.out";
    float *raw_wave = IOHandler::load_to_array(wave_path);

    // chunk into short time windows
    float ** frames = frame(raw_wave, WAVEFORM_LENGTH, WIN_LENGTH, HOP_LENGTH, N_FFT);
    IOHandler::write(frames, NUM_FRAMES, N_FFT, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/arduino/orig_frames.txt");

    // stft + spectrum transformation
    Complex ** stft_frames = stft(frames);
    float ** spec_frames = to_energy(stft_frames, NUM_FRAMES, N_FFT / 2 + 1);
    IOHandler::write(spec_frames, NUM_FRAMES, N_FFT / 2 + 1,"/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/arduino/spec_frames.txt");

    float ** bank = filter_bank(40);
    IOHandler::write(bank, 40, 1 + N_FFT/ 2, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/arduino/filter_bank.txt");
    float ** fb = dot_product(bank, transpose(spec_frames, NUM_FRAMES, 1 + N_FFT/2), 40, 1+N_FFT/2, 1+N_FFT/2, NUM_FRAMES);

    log_magnitude(fb, N_FILTER, NUM_FRAMES);
    IOHandler::write(fb, N_FILTER, NUM_FRAMES, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/arduino/filter_banks.txt");

    // mfcc

}