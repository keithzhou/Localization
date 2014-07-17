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

#define TIMING_SAMPLING_INTERVAL_MICRO  10
#define BUFFER_SIZE 2048 * 13
#define INDEX_TRIGGER BUFFER_SIZE/5
#define BUFFER_SIZE_DELAY 500

unsigned short buffer_delay[BUFFER_SIZE_DELAY];
CircularBuffer buffer_mic1(BUFFER_SIZE);
CircularBuffer buffer_mic2(BUFFER_SIZE);

int i = 0;
ADC *adc = new ADC(); // adc object

const int channelA2 = ADC::channel2sc1aADC0[2];
const int channelA3 = ADC::channel2sc1aADC1[3];
const int channelA10 = ADC::channel2sc1aADC1[10];
const int channelA11 = ADC::channel2sc1aADC0[11];

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


// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  pinMode(A2, INPUT); 
  pinMode(A3, INPUT); 
  pinMode(A10, INPUT); 
  pinMode(A11, INPUT); 
  
  highSpeed8bitADCSetup();
}

byte value1;
byte value2;
byte value3;
byte value4;
unsigned short sensorValue;
// the loop routine runs over and over again forever:
void loop() {
  // note that all code must be inside the following if to avoid jitters on sampling interval
  if (sinceLastRead >= TIMING_SAMPLING_INTERVAL_MICRO) {
    sinceLastRead -= TIMING_SAMPLING_INTERVAL_MICRO;
    highSpeed8bitAnalogReadMacro(channelA2,channelA3,value1,value2);
    highSpeed8bitAnalogReadMacro(channelA11,channelA10,value3,value4);
    buffer_mic1.add(value1);
    buffer_mic2.add(value2);
//    buffer_mic1.add(value3);
//    buffer_mic2.add(value4);
    
    if (buffer_mic1.get_current_length() == BUFFER_SIZE && abs(buffer_mic1.get_data_at_index(INDEX_TRIGGER) - 130) > 20) {
      Serial.println("START");
      for (int j = 0; j < BUFFER_SIZE; j++) {
        Serial.print(buffer_mic1.get_data_at_index(j));
        Serial.print(" ");
        Serial.print(buffer_mic2.get_data_at_index(j));
        Serial.print("\n");
      }
      Serial.println("END");
      buffer_mic1.clear_buffer();
      buffer_mic2.clear_buffer();
    }
  
//    buffer_delay[i++] = sinceLastRead;    
//    if (i >= BUFFER_SIZE_DELAY) i -= BUFFER_SIZE_DELAY;
//    if (sinceStart > 1e6 * 5) {
//      for (int j = 0; j < BUFFER_SIZE_DELAY; j++) {
//        Serial.println(buffer_delay[j]);
//      }
//      sinceStart = 0;
//    }
  }
}
