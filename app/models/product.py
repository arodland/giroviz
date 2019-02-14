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

