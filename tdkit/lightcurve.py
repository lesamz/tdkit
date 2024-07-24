import numpy as np
from astropy.time import Time

class LightCurve:
    """ 
    A class to store light curve data,

    Attributes:
        time (astropy Time object): time array
        mag (np.ndarray): magnitude array
        mag_err (np.ndarray): magnitude error array
    """
    def __init__(self, time, mag, mag_err=None):
        """
        Initializes the LightCurve object.

        Args:
            time (astropy Time object): time array
            mag (np.ndarray): magnitude array
            mag_err (np.ndarray): magnitude error array
        """

        if isinstance(time, Time): self.time = time
        else: raise TypeError('time must be an astropy Time object')
        
        self.mag = mag

        if mag_err is None: self.mag_err = np.zeros_like(mag)
        else: self.mag_err = mag_err