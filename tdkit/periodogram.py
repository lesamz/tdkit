import numpy as np

from astropy.timeseries import LombScargle
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

class Periodogram:
    """
    A class to represent and compute periodograms.

    Attributes:
        time (astropy.time.Time): The time array.
        yarray (np.ndarray): The y array.
        yarray_err (np.ndarray): The y error array.
        ls (LombScargle): The LombScargle object.
        xfrqs (Quantity): The periodogram's frequency array .
        xpers (Quantity): The periodogram's period array.
        ypower (np.ndarray): The periodogram's power array.
    """

    def __init__(self, time, yarray, yarray_err=None, cadence='auto'):
        """
        Initializes the Periodogram

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

        self.cadence = self.get_cadence(self.time, cadence)
        self.ow_length = self.get_window_length(self.time)
        self.peak_err_ow_frq = self.get_window_error(self.ow_length)


    def compute(self, frqlims='auto', frqgrid='auto', fals=np.array([0.05]), samples_per_peak=100):
        """
        Compute the periodogram.

        Args:
            frqlims (tuple or str, optional): The frequency limits. Defaults to 'auto'.
            fals (np.ndarray, optional): False alarm levels. Defaults to np.array([0.05]).
        """

        self.samples_per_peak = samples_per_peak

        self.ls = LombScargle(self.time, self.yarray, self.yarray_err)
        
        # Determine the frequency limits
        if frqlims == 'auto':
            self.minimum_frequency, self.maximum_frequency = self.get_frequency_limits(self.ow_length, self.cadence)
        elif isinstance(frqlims, tuple):
            if (isinstance(frqlims[0], Quantity)) and (isinstance(frqlims[1], Quantity)):
                self.minimum_frequency = frqlims[0]
                self.maximum_frequency = frqlims[1]
        else: raise TypeError('frqlims must be a tuple of astropy Quantity objects or "auto"')

        # Determine the frequency grid
        if frqgrid == 'auto':
            # self.xfrqs = self.ls.autofrequency(samples_per_peak=samples_per_peak,
                                            # minimum_frequency=self.minimum_frequency,
                                            # maximum_frequency=self.maximum_frequency)
            # print(self.xfrqs[0], self.xfrqs[-1], len(self.xfrqs))
            self.xfrqs = self.get_frequency_grid(self.ow_length, self.minimum_frequency, self.maximum_frequency, samples_per_peak=self.samples_per_peak)
            # print(self.xfrqs[0], self.xfrqs[-1], len(self.xfrqs))
        elif isinstance(frqgrid, Quantity):  
            self.xfrqs = frqgrid 
        else: raise TypeError('frqgrid must be an astropy Quantity object or "auto"')


        self.ypower = self.ls.power(frequency=self.xfrqs)
        self.xpers = 1 / self.xfrqs

        self.faps_baluev = self.ls.false_alarm_probability(self.ypower, 
                                                           samples_per_peak=self.samples_per_peak, 
                                                           minimum_frequency=self.minimum_frequency,
                                                           maximum_frequency=self.maximum_frequency,
                                                           method='baluev')
        self.fals_baluev = self.ls.false_alarm_level(fals, 
                                                     samples_per_peak=self.samples_per_peak, 
                                                     minimum_frequency=self.minimum_frequency,
                                                     maximum_frequency=self.maximum_frequency,
                                                     method='baluev')

    
    @staticmethod
    def get_cadence(time: Time, cadence: str = 'auto'):
        """
        Determine the cadence of the observations.

        Args:
            cadence (Quantity or str, optional): The cadence. Defaults to 'auto'.
        """
        if cadence == 'auto': cadence = np.median(np.diff(time.jd)) * u.d
        elif isinstance(cadence, Quantity): cadence = cadence
        else: raise TypeError('cadence must be an astropy Quantity object or "auto"')

        return cadence

    @staticmethod
    def get_window_length(time: Time):
        """
        Determine the length of the observing window.
        """

        ow_length = (time.jd.max() - time.jd.min()) * u.d
        
        return ow_length
    
    @staticmethod 
    def get_window_error(window_length: Quantity):
        """
        Determine the error of the observing window.
        """
        peak_err_ow_frq = (1 / window_length) / 2

        return peak_err_ow_frq

    @staticmethod
    def get_frequency_limits(ow_length: Quantity, cadence: Quantity, frqlims='auto'):
        """
        Determine the frequency limits for the periodogram.
        """
        minimum_frequency = 1 / ow_length * 2
        maximum_frequency = 1 / (2 * cadence)

        return minimum_frequency, maximum_frequency
    
    @staticmethod 
    def get_frequency_grid(window_length: Quantity, minimum_frequency: Quantity, maximum_frequency: Quantity, samples_per_peak: float):

        df = 1 / window_length / samples_per_peak

        n_elements = ((maximum_frequency - minimum_frequency) / df).si

        Nf = 1 + int(np.round(n_elements))

        xfrqs = minimum_frequency + df * np.arange(Nf)

        return xfrqs
        

    # @staticmethod
    # def get_frequency_grid(ow_length, minimum_frequency, maximum_frequency, samples_per_peak=100):


    def find_peaks(self, method='max'):
        """
        Find the peaks in the periodogram.

        Args:
            method (str, optional): The method to find peaks. Defaults to 'max'.
        """
        if method == 'max':
            self.peak_xper, self.peak_xfrq, self.peak_y, self.peak_fap = self.find_peak_max(self.xpers, self.xfrqs, self.ypower, self.faps_baluev)
            self.peak_err_ow_per = self.transform_peakerror_frq2per(self.peak_xper, self.peak_err_ow_frq)


    @staticmethod
    def find_peak_max(xpers, xfrqs, yvalues, faps_baluev):
        """
        Find the peak with the maximum power.

        Args:
            xpers (Quantity): The period array.
            xfrqs (Quantity): The frequency array.
            yvalues (np.ndarray): The power array.
            faps_baluev (np.ndarray): The false alarm probabilities.

        Returns:
            tuple: The period, frequency, power, and false alarm probability of the peak with the maximum power.
        """
        peak_arg = np.argmax(yvalues) 
        peak_xper = xpers[peak_arg]
        peak_xfrq = xfrqs[peak_arg]
        peak_y = yvalues[peak_arg]
        peak_fap = faps_baluev[peak_arg] 
        
        return peak_xper, peak_xfrq, peak_y, peak_fap

    @staticmethod        
    def transform_peakerror_frq2per(peak_xper, peak_err_ow_frq):
        """
        Transform peak error from frequency to period space.

        Args:
            peak_xper (Quantity): The period of the peak.
            peak_err_ow_frq (Quantity): The half peak width in frequency space.

        Returns:
            Quantity: The half peak width in period space
        """
        peak_centr_per = peak_xper 
        peak_centr_frq = 1 / peak_centr_per
        peak_left_per = 1 / (peak_centr_frq + peak_err_ow_frq) 
        peak_right_per = 1 / (peak_centr_frq - peak_err_ow_frq) 
        # Uncertainty is the mean of both half peak widths
        peak_err_ow_per = np.mean([peak_right_per.value-peak_centr_per.value, peak_centr_per.value-peak_left_per.value]) * peak_xper.unit
        
        return peak_err_ow_per

            
 
    
