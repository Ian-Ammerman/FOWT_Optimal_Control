import numpy as np
from scipy.signal import convolve
from scipy import signal


def get_PSD_limited(time, signal, s, F_low, F_hgh):
    """
    Function for obtaining power spectral densities of a time series with lower and upper frequency limits.

    Inputs:
    - time: Time vector
    - signal: Variable (time series data)
    - s: Smoothing parameter
    - F_low: Lower frequency limit
    - F_hgh: Upper frequency limit

    Outputs:
    - F: Frequency range (Hz) within the specified limits
    - PSD: Power density spectrum (variable units^2/Hz) within the specified frequency limits
    """
    # Compute PSD for the entire frequency range
    F, PSD = get_PSD(time, signal, s)

    # Filter frequencies and PSD values within the specified limits
    F_of_int = (F > F_low) & (F < F_hgh)
    F = F[F_of_int]
    PSD = PSD[F_of_int]

    return F, PSD


def get_PSD(xi, yi, nsmooth):
    """
    Function for obtaining power spectral densities of a time series.

    Inputs:
    - xi: Time vector
    - yi: Variable (time series data)
    - nsmooth: Smoothing parameter (larger number = more averaging, 1 = no averaging, typically between 5-25)

    Outputs:
    - f: Frequency range (Hz)
    - PSD: Power density spectrum (variable units^2/Hz)
    """
    # Number of data points
    npoints = len(xi)

    # Determine sampling rate
    begintime = xi[0]
    endtime = xi[-1]
    dt = (endtime - begintime) / (npoints - 1)
    fsample = 1 / dt

    # Compute Fourier Transform of yi
    Pf = np.fft.fft(yi)

    # Compute the PSD of yi
    Pd = 2 * np.abs(Pf) ** 2 / npoints / fsample

    # Create vector of frequencies associated with PSD data
    nc = len(Pd)
    f = np.linspace(fsample / nc, fsample, nc)

    # Smooth PSD
    window = np.ones(nsmooth) / nsmooth  # Vector used in convolution smoothing
    PSD = convolve(Pd, window, mode='same')  # Smoothing - convolution of Pd and window vector

    return f, PSD


def get_RAO(time, signal_data, elev_data, s, F_low, F_hgh, nperseg=1024, input_threshold=0.1):
    # Find the credible limits based on the input data `elev_data`:
    f, psd_wave = get_PSD_limited(time, elev_data, s, F_low, F_hgh)
    psd_peak = psd_wave.max()
    f_valid = f[psd_wave > input_threshold * psd_peak]
    F_low = f_valid[0]
    F_hgh = f_valid[-1]
    np_ = time.shape[0]
    begin_time = time[0]
    end_time = time[-1]
    dt = (end_time - begin_time) / (np_ - 1)
    fsample = 1 / dt
    f, crs = signal.csd(signal_data, elev_data, fs=fsample, nperseg=nperseg)
    _, ato = signal.csd(elev_data, elev_data, fs=fsample, nperseg=nperseg)

    rao = np.abs(crs / ato)

    # Smooth PSD
    window = np.ones(s) / s  # Vector used in convolution smoothing
    rao = convolve(rao, window, mode='same')  # Smoothing

    # Filter frequencies and PSD values within the specified limits
    F_of_int = (f > F_low) & (f < F_hgh)
    f = f[F_of_int]
    rao = rao[F_of_int]

    x = 1/f
    y = rao
    return x, y
