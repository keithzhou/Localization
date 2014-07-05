/*
  AnalogReadSerial
  Reads an analog input on pin 0, prints the result to the serial monitor.
  Attach the center pin of a potentiometer to pin A0, and the outside pins to +5V and ground.
 
 This example code is in the public domain.
 */
elapsedMicros sinceLastRead;
elapsedMicros sinceStart;

#define TIMING_SAMPLING_INTERVAL_MICRO  50
#define BUFFER_SIZE 2048

int buffer_delay[BUFFER_SIZE];
int buffer_mic1[BUFFER_SIZE];
int i = 0;

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

// the loop routine runs over and over again forever:
void loop() {
  if (sinceLastRead > TIMING_SAMPLING_INTERVAL_MICRO) {
    int sensorValue = analogRead(A0);
    //Serial.println(sensorValue);
    buffer_delay[i] = sinceLastRead;
    buffer_mic1[i] = sensorValue;
    i++;
    
    if (i >= BUFFER_SIZE) i -= BUFFER_SIZE;
    
    sinceLastRead -= TIMING_SAMPLING_INTERVAL_MICRO;
  }
  if (sinceStart > 1e6 * 20) {
    for (int j = 0; j < BUFFER_SIZE; j++) {
      Serial.println(buffer_delay[j]);
    }
    sinceStart = 0;
  }
}
