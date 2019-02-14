import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm

def real_sph(m, n, theta, phi):
    if m == 0:
        return np.real(scipy.special.sph_harm(m, n, theta, phi))
    else:
        harm = scipy.special.sph_harm(abs(m), n, theta, phi)
        if m > 0:
            harm = np.real(harm)
        else:
            harm = np.imag(harm)

        odd_even = -1 if m % 2 else 1
        return np.sqrt(2) * odd_even * harm

def longitude_to_theta(longitude):
    return longitude * np.pi / 180.

def latitude_to_phi(latitude):
    return (latitude + 90) * np.pi / 180.

class SPHModel:
    def __init__(self, order=3, L1_wt=0.6):
        self.order = order
        self.L1_wt = L1_wt

    def train(self, df, z):
        sph = []
        alpha = []

        theta = longitude_to_theta(df['station.longitude'].values)
        phi = latitude_to_phi(df['station.latitude'].values)
        confidence = df.cs.values

        for n in range(self.order):
            for m in range(-n, n+1):
                sph.append(real_sph(m, n, theta, phi).reshape(-1, 1))
                alpha.append(0 if n == 0 else 0.005)
#                alpha.append(0.08 * n)
        sph = np.hstack(sph)

        self.wls_model = sm.WLS(z, sph, confidence)
        self.wls_fit = self.wls_model.fit_regularized(alpha=np.array(alpha), L1_wt=self.L1_wt, refit=True)

    def predict(self, longitude, latitude, weight=1):
        pred = np.zeros(longitude.shape)

        coeff = self.wls_fit.params
        coeff_idx = 0
        for n in range(self.order):
            for m in range(-n, n+1):
                sh = real_sph(m, n, longitude_to_theta(longitude), latitude_to_phi(latitude))
                wt = 1 if n == 0 else weight
                pred = pred + wt * coeff[coeff_idx] * sh
                coeff_idx = coeff_idx + 1

        return pred
