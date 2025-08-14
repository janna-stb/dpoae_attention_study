import os
import numpy as np
import matplotlib.pyplot as plt
from cross_correlation import cross_corr

def stimuli_correlation(ndp, dp, stim1, stim2, filename=None, fs=44100):
    """Calculates the cross-correlation between the stimuli and the distortion product waveform.

    Parameters
    ----------
    ndp : int
        The harmonic number of the distortion product
    dp : array
        The distortion product waveform
    stim1 : array
        The first stimulus waveform
    stim2 : array
        The second stimulus waveform
    filename : str
        The filename (including the path) where the plots should be saved
    fs : int
        The sampling frequency

    Returns
    -------
    None
    """
    # if directory for plots does not exist, create it
    if filename is not None:
        plot_dir = os.path.dirname(filename)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    stim_list = [stim1, stim2, stim1 + stim2]
    plt.figure(figsize=(10, 15))

    for i, stim in enumerate(stim_list):    
        lag, corr_real, corr_imag, env = cross_corr(dp, stim)
        lag = lag / fs * 1000

        # find peak amplitude between -15 and 15 ms
        border = 15
        idx_min = np.argmin(np.abs(lag + border))
        idx_max = np.argmin(np.abs(lag - border))
        env_range = env[idx_min:idx_max + 1]
        mag, idx = np.max(env_range), np.argmax(env_range)
        idx = idx + idx_min
        lag_peak = lag[idx]

        # determine if peak is significant
        idx_noise = np.where(((lag < -70) & (lag > -750)) | ((lag > 70) & (lag < 750)))[0]
        env_noise = env[idx_noise]
        perc95 = np.percentile(env_noise, 95)
        if mag > perc95:
            sig_res = True
        else:
            sig_res = False
                        
        # plot the correlations
        plt.subplot(3, 1, i+1)
        plt.plot(lag, corr_real, label='real')
        plt.plot(lag, corr_imag, label='imag')
        plt.plot(lag, env, label='envelope', c='k')
        plt.plot(lag, -env, c='k')
        if sig_res:
            plt.axvline(lag_peak, c='k', ls='--', label=f'Sig. peak at {lag_peak:.2f} ms')
        plt.legend(loc = 'upper right', ncol=2)
        plt.xlabel('Lag [ms]')
        plt.ylabel('Correlation')
        if i < 2:
            plt.title(f'Correlation of DP {ndp} vs. f{i+1}')
        else:
            plt.title(f'Correlation of DP {ndp} vs. f1 + f2')
        plt.grid()
        plt.xlim(-100, 100)
        plt.ylim(-max(env), max(env))
        plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        plt.close()

    return sig_res