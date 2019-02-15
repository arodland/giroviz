import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

class IRISplineModel:
    def __init__(self, filename):
        self.filename = filename

    def train(self, metric):
        data = pd.read_fwf(self.filename, sep='\s+', header=None, names=["lat", "lon", "nmf2", "fof2", "md", "mufd"])
        z = data[metric].values.reshape((46, 91))
        lat = np.linspace(-90, 90, 46)
        lon = np.linspace(-180, 180, 91)
        self.spline = RectBivariateSpline(lat, lon, z)

    def predict(self, longitude, latitude, weight=None):
        return self.spline(latitude.flatten(), longitude.flatten(), grid=False).reshape(longitude.shape)
        
