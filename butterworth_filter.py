import scipy.signal as sig

def butter_filter(x, cutoff, btype, output='sos', order=3, fs=44100):
    """Applies a Butterworth filter to the signal.

    Parameters
    ----------
    x : array
        The signal to be filtered
    cutoff : float
        The cutoff frequency. If btype is 'bandpass' or 'bandstop', cutoff should be a list with two values.
    btype : str
        The type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
    output : str
        The output type ('ba', 'sos')
    order : int
        The order of the filter
    fs : int
        The sampling frequency

    Returns
    -------
    x_filt : array
        The filtered signal
    """

    if output == 'ba':
        b, a = sig.butter(order, cutoff, btype=btype, output='ba', fs=fs)
        x_filt = sig.filtfilt(b, a, x)
    elif output == 'sos':
        sos = sig.butter(order, cutoff, btype=btype, output='sos', fs=fs)
        x_filt = sig.sosfiltfilt(sos, x)
    
    return x_filt

def filterA(f):
    """Applies the A-weighting filter to the signal.

    Parameters
    ----------
    f : array 
        The frequency values
    
    Returns
    -------
    A : array
        The A-weighting filter
    """
    c1 = 3.5041384e16
    c2 = 20.598997**2
    c3 = 107.65265**2
    c4 = 737.86223**2
    c5 = 12194.217**2

    f = f**2
    num = c1 * f**4
    den = (f + c2)**2 * (f + c3) * (f + c4) * (f + c5)**2
    A = num / den
    return A  