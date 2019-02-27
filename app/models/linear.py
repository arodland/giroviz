class LinearModel:
    def __init__(self, base_model, scale, offset):
        self.base_model = base_model
        self.scale = scale
        self.offset = offset

    def predict(self, longitude, latitude):
        return self.scale * self.base_model.predict(longitude, latitude) + self.offset


