import data
import datetime as dt
from datetime import timezone
import sys
import numpy as np
from models import SPHModel, GP3DModel, LogSpaceModel, HybridModel
from plot import Plot

metric = sys.argv[1]

df = data.get_data()
df = data.filter(df, 
        max_age = dt.datetime.now()-dt.timedelta(minutes=60),
        required_metrics = [metric],
        min_confidence = 0.01
        )

model = LogSpaceModel(
        HybridModel(SPHModel(), 0.8, GP3DModel(), 0.9)
    )

model.train(df, df[metric].values)
pred = model.predict(df['station.longitude'].values, df['station.latitude'].values)

error = pred - df[metric].values
print(df[metric].values)
print(pred)
print(error)
print(np.sqrt(np.mean(error ** 2)))

plt = Plot(metric, dt.datetime.now(timezone.utc))
if metric == 'mufd':
    plt.scale_mufd()
else:
    plt.scale_generic()

plt.draw_contour(model)
plt.draw_dots(df, metric)
plt.draw_title(metric)
plt.write('/output/%s.svg' % (metric))
plt.write('/output/%s.png' % (metric))
