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
const int NUM_FRAMES = 118;                 // assumes 1.2 seconds of audio; 25ms window; 10ms hop

/**
 * Frame the audio into overlapping windows, padding with zeros
 * to ensure each window is of length >= N_FFT.
 */
void hamming(float window[], int window_size) {
    for(int i = 0; i < window_size; i++) {
        window[i] *= 0.54 - (0.46 * cos( (2 * PI * i) / (window_size - 1) ));
    }
};

float ** frame(float waveform[], int win_length, const int hop_length, const int nfft) {

  float** frames = new float*[NUM_FRAMES];

  bool pad = nfft > win_length;
  int frame_length = pad ? nfft : win_length;
  int offset = pad ? (nfft - win_length) / 2 : 0;
  int start = 0;  

  for (int i = 0; i < NUM_FRAMES; i++) {

    float* frame = new float[frame_length];

    for (int j = 0; j < offset; j++) frame[j] = 0.0;
    for (int k = 0; k < win_length; k++) {
      frame[offset+k] = waveform[start+k];
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
    Complex* left_frame = new Complex[frame_length / 2];
    for (int k = 0; k < frame_length / 2; k++) {
      left_frame[k] = stft_frame[k];
    }

    stft_frames[i] = left_frame;
  }

  return stft_frames;
};

float ** magnitude(Complex ** stft_frames, int num_frames = NUM_FRAMES, int frame_length = N_FFT) {

  float** mag_frames = new float*[num_frames];

  for (int i = 0; i < num_frames; i++) {
    float* mag_frame = new float[frame_length];
    for (int j = 0; j < frame_length; j++) {
      mag_frame[j] = stft_frames[i][j].absolute_value();
    }
    mag_frames[i] = mag_frame;
  }

  return mag_frames;
}

/**
 * Convert the power / magnitude spectrum into Filter banks (spectrogram), which
 * are the input features to our model.
 */
float * mel_filters(int nfilter, int sr = SIGNAL_RATE) {
  float low_freq = 0.0;
  float high_freq = (2595 * std::log10(1 + (sr/2)/ 700.0f));
  float step = (high_freq - low_freq) / (nfilter+1);
  
  float filters [nfilter+2];
  filters[0] = 0.0f;
  for (int i = 1; i < nfilter+2; i++) {
      filters[i] = filters[i-1] + step;
  }
  return filters;
}

void mel_to_hz() {};

void magnitude_to_db() {};

void power_to_db() {};

//float ** filter_banks() {};


/**
 * Test main()
 */
int main() {

    std::string wave_path = "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/sample_wave.out";
    float *raw_wave = IOHandler::load_to_array(wave_path);

    // chunk into short time windows
    float ** frames = frame(raw_wave, WIN_LENGTH, HOP_LENGTH, N_FFT);
    IOHandler::write(frames, NUM_FRAMES, N_FFT, "/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/arduino/orig_frames.txt");

    // stft + magnitude transformation
    Complex ** stft_frames = stft(frames);
    float ** mag_frames = magnitude(stft_frames);
    IOHandler::write(mag_frames, NUM_FRAMES, N_FFT / 2,"/home/thomas/Dir/ccny/ccny-masters-thesis/cpp/out/arduino/stft_magnitude_frames.txt");

    // filter banks

}