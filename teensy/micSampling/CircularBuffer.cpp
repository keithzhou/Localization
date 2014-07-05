CircularBuffer::CircularBuffer(int buffeer_size)
{
    _buffer = (int *)malloc(sizeof(int) * buffer_size);
    _buffer_size = buffer_size;
}

