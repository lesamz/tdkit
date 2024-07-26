import numpy as np

from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
import astropy.units as u



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

