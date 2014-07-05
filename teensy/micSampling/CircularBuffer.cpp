#include "CircularBuffer.h"

CircularBuffer::CircularBuffer(int buffer_size)
{
    _buffer = (byte *)malloc(sizeof(byte) * buffer_size);
    _buffer_size = buffer_size;
    _index_current = 0;
    _length_current = 0;
}

int CircularBuffer::get_data_at_index(int i) {
  return _buffer[(_index_current + i) % _buffer_size];
}

void CircularBuffer::add(int i) {
  _buffer[_index_current] = i;
  _index_current ++;
  if (_length_current < _buffer_size)
    _length_current ++;
  if (_index_current >= _buffer_size)
    _index_current = 0;
}

int CircularBuffer::get_current_length() {
  return _length_current;
}

void CircularBuffer::clear_buffer() {
  _length_current = 0;
}


void CircularBuffer::print_buffer() {
  for (int i = 0; i < _buffer_size; i++) {
    Serial.println(get_data_at_index(i));
  }
}
