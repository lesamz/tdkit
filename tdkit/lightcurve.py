class LightCurve:
    def __init__(self, time, flux, fluxerr):
        self.time = time
        self.flux = flux
        self.flux_err = fluxerr