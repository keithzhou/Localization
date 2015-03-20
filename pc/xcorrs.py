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
    s4 = (sig4 - np.mean(sig4))/np.std(sig4)
    xcorr12 = xcorr_freq(s1,s2,SAMPLING_RATE, doPhaseTransform, doBandpassFiltering)
    xcorr13 = xcorr_freq(s1,s3,SAMPLING_RATE, doPhaseTransform, doBandpassFiltering)
    xcorr14 = xcorr_freq(s1,s4,SAMPLING_RATE, doPhaseTransform, doBandpassFiltering)
    xcorr23 = xcorr_freq(s2,s3,SAMPLING_RATE, doPhaseTransform, doBandpassFiltering)
    xcorr24 = xcorr_freq(s2,s4,SAMPLING_RATE, doPhaseTransform, doBandpassFiltering)
    xcorr34 = xcorr_freq(s3,s4,SAMPLING_RATE, doPhaseTransform, doBandpassFiltering)
    return (xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34)

def xcorr_freq(s1,s2,SAMPLING_RATE, doPhaseTransform = True, doBandpassFiltering = True, ): 
    assert(len(s1)==len(s2))
    pad = len(s1) * 2 - 1
    lenOld = len(s1)
    s1 = np.hstack([s1,np.zeros(pad)])
    s2 = np.hstack([s2,np.zeros(pad)])
    f_s1 = fft(s1)
    f_s2 = fft(s2)

    W = fftfreq(s1.size, 1.0 / SAMPLING_RATE)
    bolla = (abs(W) < .5e3) | (abs(W) > 7e3)
    if doBandpassFiltering:
        f_s1[bolla] = 0.0
        f_s2[bolla] = 0.0

    f_s1c = np.conj(f_s1)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c 
    denom = np.abs(f_s)
    denom[denom < 1e-5] = 1e-5
    if doPhaseTransform == True:
        f_s = f_s / denom  
    assert(np.any(np.isnan(f_s))==False)

    td = ifft(f_s)
    pos = td[:lenOld]
    neg = td[-lenOld+1:]
    return np.real(np.hstack([neg,pos]))
