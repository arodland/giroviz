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
