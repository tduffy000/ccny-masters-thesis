#include <Complex.h>

#include <TensorFlowLite.h>
#include <Stepper.h>

/**
 * Feature Engineering
 */
const int SIGNAL_RATE = 16000;
const int WIN_LENGTH = SIGNAL_RATE * 0.025;
const int HOP_LENGTH = SIGNAL_RATE * 0.01; 
const int N_FFT = 512;
const int WAVEFORM_LENGTH = 16000 * 1.2; // sr * seconds
const int NUM_FRAMES = 98; // assumes 1.2 seconds of audio; 25ms window; 10ms hop

/**
 * Frame the audio into overlapping windows, padding with zeros
 * to ensure each window is of length >= N_FFT.
 */

void hamming(float window[], int window_size) {
    for(int i = 0; i < window_size; i++) {
        window[i] *= 0.54 - (0.46 * cos( (2 * PI * i) / (window_size - 1) ));
    }
};
 
float ** frame(float waveform[], int win_length, int hop_length, int nfft) {

  bool pad = nfft > win_length;
  int frame_length = pad ? nfft : win_length;
  float *frames[NUM_FRAMES];

  int offset = 0;
  for (int i = 0; i < NUM_FRAMES; i++) {
    
    float frame[frame_length];
    int start = pad ? (nfft - win_length) / 2 : 0; // TODO: test this
    for (int j = start; j < win_length; j++) {
      frame[j] = waveform[offset+j];
      offset += hop_length;
    }
    hamming(frame, nfft);
    frames[i] = frame;
  }
  return frames;
}

/**
 * Perform the Short-term Fourier transform on each of the windows
 * which we framed above. Then take the magnitude / power of that transformation.
 */
void fft(float x[], int size) {};

float ** stft(float windows[][N_FFT], int num_frames = NUM_FRAMES, int frame_length = N_FFT) {
  float *stft_frames[NUM_FRAMES];

  // the input windows have already been zero-padded to have N_FFT length
  // and had the hamming window applied
  for (int i = 0; i < num_frames; i++) {
    
    float stft_frame[frame_length];
    for (int j = 0; j < frame_length; j++) {
      // cast to complex  
    }
    fft(stft_frame, frame_length);
    // take only the real (LH) side; b/c real-valued signnal
  }
  // recall here we only need the LHS side, b/c for real-valued signals it's reflection symmetric
  
  return stft_frames;
};

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
