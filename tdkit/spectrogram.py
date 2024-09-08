import numpy as np

# from astropy.timeseries import LombScargle
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
from astropy.time import TimeDelta

from .periodogram import Periodogram


class Spectrogram:
    """
    A class to represent and compute spectrograms.

    Attributes:
        time (astropy.time.Time): The time array.
        yarray (np.ndarray): The y array.
        yarray_err (np.ndarray): The y error array.
        ls (LombScargle): The LombScargle object.
    """

    def __init__(self, time, yarray, yarray_err=None, cadence='auto', window_length=1*u.d, window_overlap=0.2, window_filling=0.5):
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

        self.cadence = Periodogram.get_cadence(self.time, cadence)
        self.window_length = window_length.to(u.d)
        self.window_err = Periodogram.get_window_error(self.window_length)        
        self.window_overlap = window_overlap
        self.window_filling = window_filling

        self._get_windows_times()
        self._get_windows_points()
        self._get_windows_data()

    def compute(self, frqlims='auto', fals=np.array([0.05]), samples_per_peak=100):
        """
        Compute the spectrogram.

        Args:
            frqlims (tuple or str, optional): The frequency limits. Defaults to 'auto'.
            fals (np.ndarray, optional): False alarm levels. Defaults to np.array([0.05]).
        """

        self.samples_per_peak = samples_per_peak
        
        # Determine the frequency limits
        if frqlims == 'auto':
            self.minimum_frequency, self.maximum_frequency = Periodogram.get_frequency_limits(self.window_length, self.cadence)
        elif isinstance(frqlims, tuple):
            if (isinstance(frqlims[0], Quantity)) and (isinstance(frqlims[1], Quantity)):
                self.minimum_frequency = frqlims[0]
                self.maximum_frequency = frqlims[1]
        else: raise TypeError('frqlims must be a tuple of astropy Quantity objects or "auto"')

        # Determine the frequency grid
    
        self.xfrqs = Periodogram.get_frequency_grid(self.window_length, self.minimum_frequency, self.maximum_frequency, samples_per_peak=self.samples_per_peak)
        self.xpers = 1 / self.xfrqs

        self.num_grid = len(self.xfrqs)

        self.ypower_windows = np.zeros((self.num_windows, self.num_grid))
        self.faps_baluev_windows = np.zeros((self.num_windows, self.num_grid))
        self.fals_baluev_windows = np.zeros(self.num_windows)

        for i, (time_window, yarray_window, yarray_err_window) in enumerate(zip(self.list_time_windows, self.list_yarray_windows, self.list_yarray_err_windows)):
            
            periodogram_window = Periodogram(time_window, yarray_window, yarray_err_window, self.cadence)
            periodogram_window.compute(frqlims=(self.minimum_frequency, self.maximum_frequency), frqgrid=self.xfrqs, fals=fals, samples_per_peak=self.samples_per_peak)  
                         
            self.ypower_windows[i] = periodogram_window.ypower
            self.faps_baluev_windows[i] = periodogram_window.faps_baluev
            self.fals_baluev_windows[i] = periodogram_window.fals_baluev

    def peak_power_evolution(self):
        """
        Get the peak power of the spectrogram.
        """

        self.peak_power_max = np.max(self.ypower_windows, axis=1)
        self.peak_power_integrated = np.sum(self.ypower_windows, axis=1)


        self.peak_power_max_norm, self.peak_power_max_fals_norm = self.normalize_power_evolution(self.peak_power_max, self.fals_baluev_windows)
        self.peak_power_integrated_norm = self.normalize_power_evolution(self.peak_power_integrated)

    @staticmethod
    def normalize_power_evolution(ypower, false_alarm_level=None):
        ypower_off = ypower - np.min(ypower)
        ypower_norm = ypower_off / np.max(ypower_off)

        if false_alarm_level is not None:
            fal_off = false_alarm_level - np.min(ypower)
            fal_norm = fal_off / np.max(ypower_off)
            return ypower_norm, fal_norm        
        else:
            return ypower_norm

    def _get_windows_times(self):
        window_shift = TimeDelta(self.window_length * (1 - self.window_overlap))

        num_windows = int((self.time.max() - self.time.min()) / window_shift) + 1
        
        time_c_windows = self.time.min() + window_shift * np.arange(num_windows)
        
        time_i_windows = time_c_windows-self.window_length/2
        time_f_windows = time_c_windows+self.window_length/2
        time_i_windows[time_i_windows<self.time.min()] = self.time.min()
        time_f_windows[time_f_windows>self.time.max()] = self.time.max()

        self.window_shift = window_shift
        self.num_windows = num_windows
        self.time_c_windows = time_c_windows
        self.time_i_windows = time_i_windows
        self.time_f_windows = time_f_windows

    def _get_windows_points(self):
        # Create a 2D array of shape (num_windows, len(spec.time))
        time_array = self.time[:, np.newaxis]
        start_window_array = self.time_i_windows[np.newaxis, :]
        end_window_array = self.time_f_windows[np.newaxis, :]

        # Check where time values fall within the window intervals
        within_window = (time_array >= start_window_array) & (time_array <= end_window_array)

        # Sum along the time axis to get the count of points within each window
        n_points_windows = within_window.sum(axis=0)

        self.n_points_windows = n_points_windows

    def _get_windows_data(self):

        mask_windows = self.n_points_windows >= (np.max(self.n_points_windows) * self.window_filling)


        self.num_windows = np.sum(mask_windows)
        self.n_points_windows = self.n_points_windows[mask_windows]
        self.time_c_windows = self.time_c_windows[mask_windows]
        self.time_i_windows = self.time_i_windows[mask_windows]
        self.time_f_windows = self.time_f_windows[mask_windows]

        list_time_windows = []
        list_yarray_windows = []
        list_yarray_err_windows = []

        std_yarray = np.std(self.yarray)

        std_yarray_windows = []

        for ti, tf in zip(self.time_i_windows, self.time_f_windows):
            mask_data = (self.time >= ti) & (self.time < tf)
            time_window, yarray_window = self.time[mask_data], self.yarray[mask_data]
            if self.yarray_err is not None:
                yarray_err_window = self.yarray_err[mask_data]
                list_yarray_err_windows.append(yarray_err_window)
            else: 
                list_yarray_err_windows.append(None)
            list_time_windows.append(time_window)
            list_yarray_windows.append(yarray_window)
            std_yarray_windows.append(np.std(yarray_window))

        self.list_time_windows = list_time_windows
        self.list_yarray_windows = list_yarray_windows
        self.list_yarray_err_windows = list_yarray_err_windows
        self.std_yarray = std_yarray
        self.std_yarray_windows = np.array(std_yarray_windows)


        