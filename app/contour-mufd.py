import data
import datetime as dt
from datetime import timezone
import sys
import numpy as np
from models import SPHModel, GP3DModel, LogSpaceModel, HybridModel, ProductModel
from plot import Plot

df = data.get_data()
df = data.filter(df, 
        max_age = dt.datetime.now()-dt.timedelta(minutes=60),
        required_metrics = ['mufd', 'md', 'fof2'],
        min_confidence = 0.01
        )

mufd = df.mufd.values

model = ProductModel(
        LogSpaceModel( # fof2
            HybridModel(SPHModel(), 0.8, GP3DModel(), 0.9)
        ),
        LogSpaceModel( # md
            HybridModel(SPHModel(), 0.8, GP3DModel(), 0.9)
        )
    )

model.train(df, df.fof2.values, df.md.values)
pred = model.predict(df['station.longitude'].values, df['station.latitude'].values)

error = pred - mufd
print(mufd)
print(pred)
print(error)
print(np.sqrt(np.mean(error ** 2)))

plt = Plot('mufd', dt.datetime.now(timezone.utc))
plt.scale_mufd()
plt.draw_contour(model)
plt.draw_dots(df, 'mufd')
plt.draw_title('mufd')
plt.write('/output/mufd-product.svg')
plt.write('/output/mufd-product.png')
