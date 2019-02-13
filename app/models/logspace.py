import numpy as np

class LogSpaceModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def train(self, df, t):
        self.base_model.train(df, np.log(t))

    def predict(self, longitude, latitude):
        return np.exp(self.base_model.predict(longitude, latitude))

