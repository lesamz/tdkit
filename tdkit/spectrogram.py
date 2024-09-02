import numpy as np

# from astropy.timeseries import LombScargle
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
from astropy.time import TimeDelta


class Spectrogram:
    """
    A class to represent and compute spectrograms.

    Attributes:
        time (astropy.time.Time): The time array.
        yarray (np.ndarray): The y array.
        yarray_err (np.ndarray): The y error array.
        ls (LombScargle): The LombScargle object.
    """

    def __init__(self, time, yarray, yarray_err=None, cadence='auto', window=1*u.d, overlap=0.2):
        """
        Initializes the Spectrogram

        Args:
            time (astropy.time.Time): The time array.
            yarray (np.ndarray): The y array.
            yarray_err (np.ndarray, optional): The y error array. Defaults to None.
            cadence (Quantity or str, optional): The cadence. Defaults to 'auto'.
        """

        if isinstance(time, Time): self.time = time
        else: raise TypeError('time must be an astropy Time object')
        
        self.yarray = yarray
        self.yarray_err = yarray_err

        self._get_cadence(cadence)
        self.window = window.to(u.d)
        self.overlap = overlap

        self._get_windows_times()
        self._get_windows_data()


    def _get_cadence(self, cadence='auto'):
        """
        Determine the cadence of the observations.

        Args:
            cadence (Quantity or str, optional): The cadence. Defaults to 'auto'.
        """
        if cadence == 'auto': self.cadence = np.median(np.diff(self.time.jd)) * u.d
        elif isinstance(cadence, Quantity): self.cadence = cadence
        else: raise TypeError('cadence must be an astropy Quantity object or "auto"')


    def _get_windows_times(self):
        shift = TimeDelta(self.window * (1 - self.overlap))

        num_windows = int((self.time.max() - self.time.min()) / shift) + 1
        
        time_c_windows = self.time.min() + shift * np.arange(num_windows)
        
        time_i_windows = time_c_windows-self.window/2
        time_f_windows = time_c_windows+self.window/2
        time_i_windows[time_i_windows<self.time.min()] = self.time.min()
        time_f_windows[time_f_windows>self.time.max()] = self.time.max()

        self.shift = shift
        self.num_windows = num_windows
        self.time_c_windows = time_c_windows
        self.time_i_windows = time_i_windows
        self.time_f_windows = time_f_windows

    def _get_windows_data(self):
        # Create a 2D array of shape (num_windows, len(spec.time))
        time_array = self.time[:, np.newaxis]
        start_window_array = self.time_i_windows[np.newaxis, :]
        end_window_array = self.time_f_windows[np.newaxis, :]

        # Check where time values fall within the window intervals
        within_window = (time_array >= start_window_array) & (time_array <= end_window_array)

        # Sum along the time axis to get the count of points within each window
        n_points_windows = within_window.sum(axis=0)

        self.n_points_windows = n_points_windows