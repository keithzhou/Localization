/*
  AnalogReadSerial
  Reads an analog input on pin 0, prints the result to the serial monitor.
  Attach the center pin of a potentiometer to pin A0, and the outside pins to +5V and ground.
 
 This example code is in the public domain.
 */
elapsedMicros sinceLastRead;
elapsedMicros sinceStart;
#include "CircularBuffer.h"
#include <arduino.h>

#define TIMING_SAMPLING_INTERVAL_MICRO  50
#define BUFFER_SIZE 2048 * 15
#define INDEX_TRIGGER BUFFER_SIZE/3

//int buffer_delay[BUFFER_SIZE];
CircularBuffer buffer_mic1(BUFFER_SIZE);
int i = 0;
boolean isDone = false;

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

// the loop routine runs over and over again forever:
void loop() {
  int sensorValue;
  if (sinceLastRead > TIMING_SAMPLING_INTERVAL_MICRO) {
    sensorValue = analogRead(A0);
    //Serial.println(sensorValue);
    
    //buffer_delay[i] = sinceLastRead;
    buffer_mic1.add(sensorValue);
    i++;
    
    if (i >= BUFFER_SIZE) i -= BUFFER_SIZE;
    
    sinceLastRead -= TIMING_SAMPLING_INTERVAL_MICRO;
  }
  if (buffer_mic1.get_current_length() == BUFFER_SIZE && abs(buffer_mic1.get_data_at_index(INDEX_TRIGGER) - 512) > 300 && !isDone) {
    for (int j = 0; j < BUFFER_SIZE; j++) {
      Serial.println(buffer_mic1.get_data_at_index(j));
    }
    isDone = true;
  }
  if (sinceStart > 1e6 * 1000) {
    for (int j = 0; j < BUFFER_SIZE; j++) {
      Serial.println(buffer_mic1.get_data_at_index(j));
    }
    sinceStart = 0;
  }
}
