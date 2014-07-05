#ifndef CircularBuffer_h
#define CircularBuffer_h

#include "Arduino.h"
class CircularBuffer
{
    public:
        CircularBuffer(int buffer_size);
        void add(int i);
        int get_data_at_index(int i);
        void print_buffer();
        int get_current_length();
        void clear_buffer();
    private:
        int _buffer_size;
        byte *_buffer;
        int _index_current;
        int _length_current;
};
#endif
