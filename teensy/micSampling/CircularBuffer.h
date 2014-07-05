#ifndef CircularBuffer_h
#define CircularBuffer_h

#include "Arduino.h"
class CircularBuffer
{
    public:
        CircularBuffer(int buffer_size);
        add(int i);
        print_buffer();
    private:
        int _buffer_size;
        int *_buffer;
        int index_start;
        int index_end;
}
#endif
