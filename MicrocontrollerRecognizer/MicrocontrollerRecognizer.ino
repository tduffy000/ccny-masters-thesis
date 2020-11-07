#include <TensorFlowLite.h>
#include <Stepper.h>

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
 * Conversions
 */

// Connect the MEMS AUD output to the Arduino A0 pin
int mic = A0;

// Variables to find the peak-to-peak amplitude of AUD output
const int sampleTime = 50; 
int micOut;

// TODO: voice activity detection to begin recording samples
const int noiseThreshold = 20;

void setup() {
  Serial.begin(9600);
}

void loop() {
   int micOutput = findPTPAmp();
   VUMeter(micOutput);   
}   

// Find the Peak-to-Peak Amplitude Function
int findPTPAmp(){
// Time variables to find the peak-to-peak amplitude
   unsigned long startTime = millis();  // Start of sample window
   unsigned int PTPAmp = 0; 

// Signal variables to find the peak-to-peak amplitude
   unsigned int maxAmp = 0;
   unsigned int minAmp = 1023;

// Find the max and min of the mic output within the 50 ms timeframe
   while(millis() - startTime < sampleTime) 
   {
      micOut = analogRead(mic);
      if( micOut < 1023) //prevent erroneous readings
      {
        if (micOut > maxAmp)
        {
          maxAmp = micOut; //save only the max reading
        }
        else if (micOut < minAmp)
        {
          minAmp = micOut; //save only the min reading
        }
      }
   }
   
  PTPAmp = maxAmp - minAmp; // (max amp) - (min amp) = peak-to-peak amplitude
  double micOut_Volts = (PTPAmp * 3.3) / 1024; // Convert ADC into voltage

  //Uncomment this line for help debugging (be sure to also comment out the VUMeter function)
  //Serial.println(PTPAmp); 

  //Return the PTP amplitude to use in the soundLevel function. 
  // You can also return the micOut_Volts if you prefer to use the voltage level.
  return PTPAmp;   
}

// Volume Unit Meter function: map the PTP amplitude to a volume unit between 0 and 10.
int VUMeter(int micAmp){
  int preValue = 0;

  // Map the mic peak-to-peak amplitude to a volume unit between 0 and 10.
   // Amplitude is used instead of voltage to give a larger (and more accurate) range for the map function.
   // This is just one way to do this -- test out different approaches!
  int fill = map(micAmp, 23, 750, 0, 10); 

  // Only print the volume unit value if it changes from previous value
  while(fill != preValue)
  {
    Serial.println(fill);
    preValue = fill;
  }
}
