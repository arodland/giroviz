import numpy as np

class HybridModel:
    def __init__(self, model1, model1_weight, model2, model2_weight):
        self.model1 = model1
        self.model1_weight = model1_weight
        self.model2 = model2
        self.model2_weight = model2_weight

    def train(self, df, t):
        self.model1.train(df, t)
        m1_pred = self.model1.predict(df['station.longitude'].values, df['station.latitude'].values, weight=self.model1_weight)
        residual = t - m1_pred
        self.model2.train(df, residual)

    def predict(self, longitude, latitude):
        model1_pred = self.model1.predict(longitude, latitude, weight=self.model1_weight)
        residual = self.model2.predict(longitude, latitude)
        return model1_pred + self.model2_weight * residual

class LinearModel:
    def __init__(self, base_model, scale, offset):
        self.base_model = base_model
        self.scale = scale
        self.offset = offset

    def predict(self, longitude, latitude):
        return self.scale * self.base_model.predict(longitude, latitude) + self.offset

class LogSpaceModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def train(self, df, t):
        self.base_model.train(df, np.log(t))

    def predict(self, longitude, latitude):
        return np.exp(self.base_model.predict(longitude, latitude))

class ProductModel:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def train(self, df, t1, t2):
        self.model1.train(df, t1)
        self.model2.train(df, t2)

    def predict(self, longitude, latitude):
        model1_pred = self.model1.predict(longitude, latitude)
        model2_pred = self.model2.predict(longitude, latitude)
        return model1_pred * model2_pred

class DifferenceModel:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def predict(self, longitude, latitude):
        model1_pred = self.model1.predict(longitude, latitude)
        model2_pred = self.model2.predict(longitude, latitude)
        return model1_pred - model2_pred
