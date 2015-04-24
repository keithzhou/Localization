from scipy.fftpack import rfft, irfft, fftfreq,fft,ifft
import numpy as np
import config

def getXcorrs(sig1,sig2,sig3,sig4,SAMPLING_RATE, doPhaseTransform = True, doBandpassFiltering = True):
    assert (len(sig1) == len(sig2))
    assert (len(sig2) == len(sig3))
    assert (len(sig3) == len(sig4))
    s1 = (sig1 - np.mean(sig1))/np.std(sig1)
    s2 = (sig2 - np.mean(sig2))/np.std(sig2)
    s3 = (sig3 - np.mean(sig3))/np.std(sig3)
    #s4 = (sig4 - np.mean(sig4))/np.std(sig4)
    # get rid of all sig 4

    f1 = getFFT(s1, SAMPLING_RATE, doBandpassFiltering)
    f2 = getFFT(s2, SAMPLING_RATE, doBandpassFiltering)
    f3 = getFFT(s3, SAMPLING_RATE, doBandpassFiltering)
    f2c = np.conj(f2)
    f3c = np.conj(f3)
    xcorr12 = do_xcorr_freq(f1,f2c,len(s1),doPhaseTransform)
    xcorr13 = do_xcorr_freq(f1,f3c,len(s1),doPhaseTransform)
    xcorr23 = do_xcorr_freq(f2,f3c,len(s1),doPhaseTransform)
    xcorr14 = None
    xcorr24 = None
    xcorr34 = None

    return (xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34)

def getFFT(s1, SAMPLING_RATE, doBandpassFiltering = True):
    pad = len(s1) - 1
    lenOld = len(s1)
    s = np.hstack([s1,np.zeros(pad)])
    f_s1 = fft(s)

    W = fftfreq(s.size, 1.0 / SAMPLING_RATE)
    bolla = (abs(W) < .5e3) | (abs(W) > 7e3)
    if doBandpassFiltering:
        f_s1[bolla] = 0.0

    return f_s1

def do_xcorr_freq(f_s1,f_s2c,lenOld,doPhaseTransform = True):
    f_s = f_s1 * f_s2c 
    if doPhaseTransform == True:
        f_s[f_s != 0] = f_s[f_s != 0] / denom[f_s != 0]  
        assert(np.any(np.isnan(f_s))==False)

    td = ifft(f_s)
    pos = td[:lenOld]
    neg = td[-lenOld+1:]
    return np.real(np.hstack([neg,pos]))
