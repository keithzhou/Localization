import numpy as np
import math

FILENAME = "output_raw.npy"

SPEED_SOUND = 340
SAMPLING_RATE = 1e5
DISTANCE_SPEAKER = .12

def xcorr(sig1, sig2):
    output = np.correlate(sig1,sig2,'full')
    return np.argmax(output) - (len(sig2) - 1) 


if __name__ == "__main__":
    def sanityCheck():
        sig1 = [0,0,0,1,0,0]
        sig2 = [0,0,0,0,1,0]
        assert(xcorr(sig1,sig2) == -1)
        sig3 = [0,1,0,0,0,0]
        assert(xcorr(sig1,sig3) == 2)
        print "success"

    def run():
        waveform = np.load(FILENAME)
        xcorrLag = xcorr(waveform[0], waveform[1])
        ratio = xcorrLag * 1.0 / SAMPLING_RATE *  SPEED_SOUND / DISTANCE_SPEAKER
        ratio = min(max(-1.0,ratio),1.0)
        angle = math.acos(ratio) / math.pi * 180
        print "lag:%d angle:%.4f" % (xcorrLag, angle)

    #sanityCheck()
    run()
