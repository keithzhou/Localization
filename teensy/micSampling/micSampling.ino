/*
  AnalogReadSerial
  Reads an analog input on pin 0, prints the result to the serial monitor.
  Attach the center pin of a potentiometer to pin A0, and the outside pins to +5V and ground.
 
 This example code is in the public domain.
 */
elapsedMicros sinceLastRead;
elapsedMicros sinceStart;
#include "CircularBuffer.h"
#include <ADC.h>
#include <arduino.h>

#define TIMING_SAMPLING_INTERVAL_MICRO  15000
#define BUFFER_SIZE_DELAY 50
#define BUFFER_SIZE 6000
#define NUM_BYTES_TO_TRANSMIT (BUFFER_SIZE*4 + 1) // end with '\n'

int buffer_delay[BUFFER_SIZE_DELAY];


ADC *adc = new ADC(); // adc object

const int channelA2 = ADC::channel2sc1aADC1[2];
const int channelA3 = ADC::channel2sc1aADC1[3];
//const int channelA10 = ADC::channel2sc1aADC1[10];
//const int channelA11 = ADC::channel2sc1aADC0[11];
const int channelA7 = ADC::channel2sc1aADC0[7];
const int channelA8 = ADC::channel2sc1aADC0[8];
#define highSpeed8bitAnalogReadMacro(channel1, channel2, value1, value2) ADC0_SC1A = channel1;ADC1_SC1A = channel2;while (!(ADC0_SC1A & ADC1_SC1A & ADC_SC1_COCO)) {} value1 = ADC0_RA;value2 = ADC1_RA;

void highSpeed8bitADCSetup(){
  
  /*
      0 ADLPC (Low-Power Configuration)
      0 ADIV (Clock Divide Select)
      0
      0 ADLSMP (Sample time configuration)
      0 MODE (Conversion mode selection) (00=8/9, 01=12/13, 10=10/11, 11=16/16 bit; diff=0/1)
      0
      0 ADICLK (Input Clock Select)
      0
  */
  ADC0_CFG1 = 0b00000000;
  ADC1_CFG1 = 0b00000000;

   /*
      0 MUXSEL (ADC Mux Select)
      0 ADACKEN (Asynchrononous Clock Output Enable)
      0 ADHSC (High-Speed Configuration)
      0 ADLSTS (Long Sample Time Select) (00=+20 cycles, 01=+12, 10=+6, 11=+2)
      0
  */
  ADC0_CFG2 = 0b00010100;
  ADC1_CFG2 = 0b00010100;
  
  /*
      0 ADTRG (Conversion Trigger Select)
      0 ACFE (Compare Function Enable)
      0 ACFGT (Compare Function Greater than Enable)
      0 ACREN (Compare Function Range Enable)
      0 ACREN (COmpare Function Range Enable)
      0 DMAEN (DMA Enable)
      0 REFSEL (Voltage Reference Selection) (00=default,01=alternate,10=reserved,11=reserved)
  */
  ADC0_SC2 = 0b00000000;
  ADC1_SC2 = 0b00000000;
 
  /*
      1 CAL (Calibration)
      0 CALF (read only)
      0 (Reserved)
      0
      0 ADCO (Continuous Conversion Enable)
      1 AVGS (Hardware Average Enable)
      1 AVGS (Hardware Average Select) (00=4 times, 01=8, 10=16, 11=32)
      1
  */
  
  ADC0_SC3 = 0b10000000;
  ADC1_SC3 = 0b10000000;

  
  // Waiting for calibration to finish. The documentation is confused as to what flag to be waiting for (SC3[CAL] on page 663 and SC1n[COCO] on page 687+688).
  while (ADC0_SC3 & ADC_SC3_CAL) {} ;
  while (ADC1_SC3 & ADC_SC3_CAL) {} ;

}

byte values[NUM_BYTES_TO_TRANSMIT];


// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(115200);
  pinMode(A2, INPUT); 
  pinMode(A3, INPUT); 
  pinMode(A10, INPUT); 
  pinMode(A11, INPUT); 
  
  highSpeed8bitADCSetup();
  for (int j = 0; j < NUM_BYTES_TO_TRANSMIT ; j ++) {
    values[j] = 'c';
  }
  for (int j = 0; j < BUFFER_SIZE_DELAY; j ++) {
    buffer_delay[j] = -1;
  }
  //values[4] = '\n';
  values[NUM_BYTES_TO_TRANSMIT - 1] = '\n';
}
int i = 0;

// the loop routine runs over and over again forever:
//void loopTest() {
//  // note that all code must be inside the following if to avoid jitters on sampling interval
//  if (sinceLastRead >= TIMING_SAMPLING_INTERVAL_MICRO) {
//    sinceLastRead -= TIMING_SAMPLING_INTERVAL_MICRO;
//    for (int k = 0; k < 600; k ++) {
//     highSpeed8bitAnalogReadMacro(channelA7,channelA2,values[1],values[0]);
//     highSpeed8bitAnalogReadMacro(channelA8,channelA3,values[2],values[3]); 
//    }
//    // sampling all 4 channels takes around 2~3 microseconds (half and half)
//
//    // teensy3.1 writes to USB buffer in FTDI chip. if you data fits into the buffer it returns very quickly
//    // teensy and FTDI hold a partially filled buffer in case you want to transmit more data. after 3ms on teensy and 8 or 16ms on FTDI, data is scheduled to transmit on USB. can ast for immediate transmit with Serial.send_now
//    // when a full or partial buffer is ready to transmit, actual transmission happens when host controller allows. Usually host controller requests scheduled transfer 1000 times per second. occurs 0 or 1 ms after transmit
//    // USB communicates at 12Mbit/s 
//    //Serial.write(values,NUM_BYTES_TO_TRANSMIT); 
//    //Serial.flush();
//    // sending 5 bytes takes around 2 microseconds (up to 4, mostly 2)
//    // sending 9 bytes takes around 2-3-5 microseconds (up to 4, mostly 2/3)
//    // sending 33 byes takes around 4-7-10 microseconds
//    // sending 129 bytes takes around 16-17-20 microseconds 
//    
//  
//    buffer_delay[i++] = sinceLastRead;    
//    if (i >= BUFFER_SIZE_DELAY) i -= BUFFER_SIZE_DELAY;
//    if (sinceStart > 1e6 * 5) {
//      for (int j = 0; j < BUFFER_SIZE_DELAY; j++) {
//        Serial.println(buffer_delay[j]);
//      }
//      sinceStart = 0;
//      sinceLastRead = 0;
//    }
//  }
//}

void loop() {
  for (int k = 0; k < BUFFER_SIZE; k ++) {
     highSpeed8bitAnalogReadMacro(channelA7,channelA2,values[k*4 + 1],values[k*4 + 0]);
     highSpeed8bitAnalogReadMacro(channelA8,channelA3,values[k*4 + 2],values[k*4 + 3]); 
  }
  int a = sinceLastRead;
  Serial.write((byte *)& a,4);
  Serial.write(values, NUM_BYTES_TO_TRANSMIT);
  sinceLastRead = 0;
}
