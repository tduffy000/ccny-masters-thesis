#include <TensorFlowLite.h>
#include <Stepper.h>

/***************************
 * Example Sketch for the SparkFun MEMS Microphone Breakout Board
 * Written by jenfoxbot <jenfoxbot@gmail.com>
 * Code is open-source, beer/coffee-ware license.
 */

/**
 * Frame audio into short windows to prepare for Fourier transform
 */
std::vector<std::vector<float>> frame(wv std::valarray<float>, winLength int, hopLength int) {
  int offset = 0;
  std::vector<std::vector<float>> frames;

  while (offset < wv.size() - winLength) {
      
      std::vector<float> frame;
      for(int i = 0; i < winLength; ++i) {
          frame.push_back(wav[offset+i]);
      }
      frames.push_back(frame);
      offset += signalHopLength;
  }
  return frames;
}

/**
 * Perform windowing  
 */
void hamming(int winLength, std::vector<&std::valarray<float>> frames) {}

/**
 * Apply the Fourier transform and then compute the Power spectrum.
 */
void stft() {}

/**
 * Apply filter banks.
 */
void filterBank() {}

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
