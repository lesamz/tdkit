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


    # def bin(self, binsize, aggregate_func=np.median):
    #     """
    #     Downsamples the light curve using the given bin size and aggregation function.

    #     Args:
    #         binsize (astropy quantity): Size of the bins.
    #         aggregate_func (function, optional): Function to aggregate the values within each bin. Defaults to np.median.
    #     """

    #     ts = TimeSeries(time=self.time, data={
    #                                             'mag': self.mag,
    #                                             'mag_err': self.mag_err
    #                                             })

    #     # Perform the downsampling
    #     downsampled_ts = aggregate_downsample(ts, time_bin_size=binsize, aggregate_func=aggregate_func)
        
    #     # Calculate the midpoint of each bin
    #     midpoints = downsampled_ts['time_bin_start'] + (downsampled_ts['time_bin_size'] / 2)

    #     binned_time = Time(midpoints, format=self.time.format, scale=self.time.scale)
    #     binned_mag = downsampled_ts['mag'].filled(np.nan)
    #     binned_mag_err = downsampled_ts['mag_err'].filled(np.nan)

    #     # Remove NaN values
    #     valid = ~np.isnan(binned_time.value) & ~np.isnan(binned_mag)
    #     binned_time = binned_time[valid]
    #     binned_mag = binned_mag[valid].data
    #     binned_mag_err = binned_mag_err[valid].data

    #     self.binned = BinnedLightCurve(binned_time, binned_mag, binsize, mag_err=binned_mag_err)

    
    def bin(self, binsize, aggregation_method='mean', error_method='weighted_mean', n_bootstrap=1000):
        """
        Downsamples the light curve using the given bin size and specified aggregation methods.

        Args:
            binsize (astropy quantity): Size of the bins.
            aggregation_method (str): Method to aggregate the magnitudes within each bin. Options are 'mean', 'median', or 'custom'.
            error_method (str): Method to calculate the error of the binned magnitudes. Options are 'weighted_mean', 'bootstrap', 'std', 'quadrature', or 'custom'.
            n_bootstrap (int, optional): Number of bootstrap samples to use when error_method is 'bootstrap'. Defaults to 1000.
        """

        # Ensure binsize is in the correct units
        if not isinstance(binsize, u.Quantity):
            raise ValueError("binsize must be an astropy Quantity with time units.")

        # Convert time to JD for calculations
        time_jd = self.time.jd

        # Determine the range of the data
        time_start = np.min(time_jd)
        time_end = np.max(time_jd)

        # Create bin edges
        bin_edges = np.arange(time_start, time_end + binsize.to(u.day).value, binsize.to(u.day).value)

        # Digitize the time data into bins
        bin_indices = np.digitize(time_jd, bin_edges) - 1  # Subtract 1 to get zero-based indices

        # Initialize lists to hold binned data
        binned_time = []
        binned_mag = []
        binned_mag_err = []

        # Loop over each bin
        for i in range(len(bin_edges) - 1):
            # Get indices of data points in the current bin
            indices_in_bin = np.where(bin_indices == i)[0]

            if len(indices_in_bin) > 0:
                # Times in the bin
                times_in_bin = time_jd[indices_in_bin]

                # Calculate the midpoint of the bin
                time_midpoint = (bin_edges[i] + bin_edges[i + 1]) / 2.0

                # Magnitudes and errors in the bin
                values = self.mag[indices_in_bin]
                errors = self.mag_err[indices_in_bin]

                # Aggregate magnitudes and calculate errors
                mag, mag_err = self._aggregate_bin(values, errors, aggregation_method, error_method, n_bootstrap)

                # Append to binned data
                binned_time.append(time_midpoint)
                binned_mag.append(mag)
                binned_mag_err.append(mag_err)

        # Convert lists to arrays
        binned_time = Time(binned_time, format='jd', scale=self.time.scale)
        binned_mag = np.array(binned_mag)
        binned_mag_err = np.array(binned_mag_err)

        # Remove NaN values
        valid = ~np.isnan(binned_mag)
        binned_time = binned_time[valid]
        binned_mag = binned_mag[valid]
        binned_mag_err = binned_mag_err[valid]

        self.binned = BinnedLightCurve(binned_time, binned_mag, binsize, mag_err=binned_mag_err)

    def _aggregate_bin(self, values, errors, aggregation_method, error_method, n_bootstrap=1000):
        values = np.atleast_1d(values)
        errors = np.atleast_1d(errors)

        N = len(values)
        if N == 0:
            return np.nan, np.nan

        if N == 1:
            # For bins with a single data point, return the value and error directly
            return values[0], errors[0]

        if aggregation_method == 'mean':
            if error_method == 'weighted_mean':
                # Weighted mean and error
                weights = 1 / errors**2
                mean = np.sum(weights * values) / np.sum(weights)
                error = np.sqrt(1 / np.sum(weights))
            elif error_method == 'quadrature':
                # Simple mean and error combined in quadrature
                mean = np.mean(values)
                error = np.sqrt(np.sum(errors**2)) / N
            elif error_method == 'std':
                # Mean and standard error
                mean = np.mean(values)
                error = np.std(values, ddof=1) / np.sqrt(N)
            else:
                raise ValueError(f"Unsupported error method '{error_method}' for aggregation method 'mean'")
            return mean, error

        elif aggregation_method == 'median':
            if error_method == 'bootstrap':
                # Median and error via bootstrapping
                median, error = self._median_with_bootstrap(values, n_bootstrap=n_bootstrap)
            elif error_method == 'std':
                # Median and standard error
                median = np.median(values)
                error = np.std(values, ddof=1) / np.sqrt(N)
            else:
                raise ValueError(f"Unsupported error method '{error_method}' for aggregation method 'median'")
            return median, error

        else:
            raise ValueError(f"Unsupported aggregation method '{aggregation_method}'")

    def _median_with_bootstrap(self, values, n_bootstrap=1000):
        medians = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(values, size=len(values), replace=True)
            medians.append(np.median(resample))
        median = np.median(values)
        error = np.std(medians)
        return median, error
    
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
