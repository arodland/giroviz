import numpy as np
import rbf
import rbf.gauss

def sph_to_xyz(lon, lat):
    lon = lon * np.pi / 180.
    lat = lat * np.pi / 180.
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z

class GP3DModel:
    def __init__(self):
        pass

    def train(self, df, t):
        x, y, z = sph_to_xyz(df['station.longitude'].values, df['station.latitude'].values)
        stdev = 0.254 - 0.158 * df.cs

        gp = rbf.gauss.gpiso(rbf.basis.mat32, (0.0, 1.0, 1.0))
        self.gp = gp_cond = gp.condition(np.vstack((x,y,z)).T, t, sigma=stdev)

    def predict(self, longitude, latitude):
        x, y, z = sph_to_xyz(longitude, latitude)
        xyz = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        pred, sd = self.gp.meansd(xyz)
        return pred.reshape(x.shape)
