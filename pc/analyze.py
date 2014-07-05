import numpy as np

FILENAME = "output_raw.npy"

def xcorr(sig1, sig2):
    output = np.correlate(sig1,sig2,'full')
    return np.argmax(output) - (len(sig2) - 1) # positive means signal 2 should be moved to the left


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
        print xcorr(waveform[0], waveform[1])

    #sanityCheck()
    run()
