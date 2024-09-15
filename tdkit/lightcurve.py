import numpy as np

from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
import astropy.units as u

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# import copy

class LightCurve:
    """ 
    A class to represent light curve data.

    Attributes:
        time (astropy Time object): Time array.
        mag (np.ndarray): Magnitude array.
        mag_err (np.ndarray): Magnitude error array.
    """

    def __init__(self, time, mag, mag_err=None):
        """
        Initializes the LightCurve object.

        Args:
            time (astropy Time object): Time array.
            mag (np.ndarray): Magnitude array.
            mag_err (np.ndarray, optional): Magnitude error array. Defaults to None.
        """        

        if isinstance(time, Time): self.time = time
        else: raise TypeError('time must be an astropy Time object')
        
        self.mag = mag

        if mag_err is None: self.mag_err = np.zeros_like(mag)
        else: self.mag_err = mag_err



    def bin(self, binsize, aggregate_func=np.median):
        """
        Downsamples the light curve using the given bin size and aggregation function.

        Args:
            binsize (astropy quantity): Size of the bins.
            aggregate_func (function, optional): Function to aggregate the values within each bin. Defaults to np.median.
        """

        ts = TimeSeries(time=self.time, data={
                                                'mag': self.mag,
                                                'mag_err': self.mag_err
                                                })

        # Perform the downsampling
        downsampled_ts = aggregate_downsample(ts, time_bin_size=binsize, aggregate_func=aggregate_func)
        
        # Calculate the midpoint of each bin
        midpoints = downsampled_ts['time_bin_start'] + (downsampled_ts['time_bin_size'] / 2)

        binned_time = Time(midpoints, format=self.time.format, scale=self.time.scale)
        binned_mag = downsampled_ts['mag'].filled(np.nan)
        binned_mag_err = downsampled_ts['mag_err'].filled(np.nan)

        # Remove NaN values
        valid = ~np.isnan(binned_time.value) & ~np.isnan(binned_mag)
        binned_time = binned_time[valid]
        binned_mag = binned_mag[valid].data
        binned_mag_err = binned_mag_err[valid].data

        self.binned = BinnedLightCurve(binned_time, binned_mag, binsize, mag_err=binned_mag_err)

    def fold(self, period, t0=None):
        """
        Folds the light curve using the given period and epoch.

        Args:
            period (astropy quantity): The period to fold the light curve on.
            t0 (astropy Time object, optional): The epoch to fold the light curve from. Defaults to the initial (minimum) time.
        """

        if t0 is None: t0 = self.time.min()

        folded_time = (self.time - t0).to(u.d) % period 
        phase = (folded_time / period).si.value

        sorted_indices = np.argsort(phase)
        sorted_phase = np.array([phase[i] for i in sorted_indices])
        sorted_mag = np.array([self.mag[i] for i in sorted_indices])
        sorted_mag_err = np.array([self.mag_err[i] for i in sorted_indices])
        sorted_time = np.array([folded_time.to(u.d).value[i] for i in sorted_indices])
        sorted_time = Time(sorted_time*u.d + t0)

        self.folded = FoldedLightCurve(sorted_time, sorted_mag, sorted_phase, period, mag_err=sorted_mag_err)



    def flatten(self, cadence=20*u.s, window_length=1*u.d, filter_scale=0.5):
        """
        Flattens the light curve using a Savitzky-Golay filter on the binned light curve.

        Args:
            window_length (astropy quantity, optional): The bin size. Defaults to 1 day.
            filter_scale (float, optional): The scaling of the bin size for the window of the Savitzky-Golay filter
        """

        lc = self.copy()
        lc.bin(window_length)

        dx = cadence.to(u.d).value

        xarray, yarray = lc.time.mjd, lc.mag
        xarray_binned, yarray_binned = lc.binned.time.mjd, lc.binned.mag

        xarray_nogaps = np.arange(xarray[0], xarray[-1]+dx, dx)

        binned_interpolator = interp1d(xarray_binned, yarray_binned, kind='linear', fill_value='extrapolate')
        yarray_nogaps_interp = binned_interpolator(xarray_nogaps)
      
        window_length_filter = int((window_length.to(u.d).value) * filter_scale / dx)
        if window_length_filter % 2 == 0: window_length_filter += 1

        yarray_nogaps_filtered = savgol_filter(yarray_nogaps_interp, window_length=window_length_filter, polyorder=3)
        
        filtered_interpolator = interp1d(xarray_nogaps, yarray_nogaps_filtered, kind='linear', fill_value="extrapolate")
        yarray_filtered = filtered_interpolator(xarray)

        flatten_time = lc.time
        flatten_mag_smooth = yarray_filtered
        flatten_mag = yarray / yarray_filtered

        if lc.mag_err is not None: flatten_mag_err = lc.mag_err / yarray_filtered
        else: flatten_mag_err = None

        self.flattened = FlattenedLightCurve(flatten_time, flatten_mag, flatten_mag_smooth, mag_err=flatten_mag_err)

    
    def copy(self):
        """
        Creates a deep copy of the LightCurve instance.

        Returns:
            LightCurve: A new instance of LightCurve with copied data.
        """
        # Create deep copies of the attributes
        new_time = self.time.copy()
        new_mag = self.mag.copy()
        new_mag_err = self.mag_err.copy()

        # Return a new instance of the class
        return self.__class__(new_time, new_mag, mag_err=new_mag_err)


class BinnedLightCurve(LightCurve):
    """
    A class to represent binned light curve data.

    Attributes:
        time (astropy Time object): Time array.
        mag (np.ndarray): Magnitude array.
        mag_err (np.ndarray): Magnitude error array.
        bins (int): Number of bins.
    """
    def __init__(self, time, mag, binsize, mag_err=None):
        """
        Initializes the BinnedLightCurve object.

        Args:
            time (astropy Time object): Time array.
            mag (np.ndarray): Magnitude array.
            binsize (astropy quantity): Size of the bins.
            mag_err (np.ndarray, optional): Magnitude error array. Defaults to None.
        """

        super().__init__(time, mag, mag_err)
        self.binsize = binsize

    # def copy(self):
    #     """
    #     Creates a deep copy of the BinnedLightCurve instance.

    #     Returns:
    #         BinnedLightCurve: A new instance of BinnedLightCurve with copied data.
    #     """
    #     # Use the base class copy method to copy common attributes
    #     copied_base = super().copy()

    #     # Copy the binsize attribute
    #     new_binsize = copy.deepcopy(self.binsize)

    #     # Return a new instance of the class with the copied attributes
    #     return self.__class__(copied_base.time, copied_base.mag, binsize=new_binsize, mag_err=copied_base.mag_err)        

class FoldedLightCurve(LightCurve):
    """
    A class to represent folded light curve data.

    Attributes:
        time (astropy Time object): Time array.
        mag (np.ndarray): Magnitude array.
        mag_err (np.ndarray): Magnitude error array.
        period (astropy quantity): The period of the folded light curve.
        phase (np.ndarray): Phase values corresponding to the folded light curve.
    """    
    
    def __init__(self, time, mag, phase, period, mag_err=None):
        """
        Initializes the FoldedLightCurve object.

        Args:
            time (astropy Time object): Time array.
            mag (np.ndarray): Magnitude array.
            phase (np.ndarray): Phase values.
            period (astropy quantity): The period of the folded light curve.
            mag_err (np.ndarray, optional): Magnitude error array. Defaults to None.
        """
        super().__init__(time, mag, mag_err)
        self.period = period
        self.phase = phase


    # def copy(self):
    #     """
    #     Creates a deep copy of the FoldedLightCurve instance.

    #     Returns:
    #         FoldedLightCurve: A new instance of FoldedLightCurve with copied data.
    #     """
    #     # Use the base class copy method to copy common attributes
    #     copied_base = super().copy()

    #     # Copy the period and phase attributes
    #     new_period = copy.deepcopy(self.period)
    #     new_phase = self.phase.copy()

    #     # Return a new instance of the class with the copied attributes
    #     return self.__class__(copied_base.time, copied_base.mag, new_phase, new_period, mag_err=copied_base.mag_err)


class FlattenedLightCurve(LightCurve):
    """
    A class to represent flattened light curve data.

    Attributes:
        time (astropy Time object): Time array.
        mag (np.ndarray): Magnitude array.
        mag_err (np.ndarray): Magnitude error array.
    """    
    def __init__(self, time, mag, mag_smooth, mag_err=None):
        """
        Initializes the FlattenedLightCurve object.

        Args:
            time (astropy Time object): Time array.
            mag (np.ndarray): Flattened Magnitude array.
            mag_err (np.ndarray, optional): Magnitude error array. Defaults to None.
            mag_smooth (np.ndarray): Original Smoothed Magnitude array.
        """
        super().__init__(time, mag, mag_err)
        self.mag_smooth = mag_smooth
