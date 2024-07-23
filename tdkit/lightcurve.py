import numpy as np
from astropy.time import Time

class LightCurve:
    def __init__(self, time, mag, magerr=None):
        if isinstance(time, Time):
            self.time = time
        else:
            raise TypeError('time must be an astropy Time object')
        
        self.mag = mag
        if magerr is None:
            self.magerr = np.zeros_like(mag)
        else:
            self.magerr = magerr