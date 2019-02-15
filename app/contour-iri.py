import data
import datetime as dt
from datetime import timezone
import sys
import numpy as np
from models import GP3DModel, IRISplineModel, HybridModel, LogSpaceModel, ProductModel
from plot import Plot

metric = sys.argv[1]

df = data.get_data()
df = data.filter(df, 
        max_age = dt.datetime.now()-dt.timedelta(minutes=60),
        required_metrics = [metric],
        min_confidence = 0.01
        )

#model = LogSpaceModel(
#        HybridModel(SPHModel(), 0.8, GP3DModel(), 0.9)
#    )

#model.train(df, df[metric].values)
irimodel = IRISplineModel("/iri.latest")
irimodel.train(metric)

pred = irimodel.predict(df['station.longitude'].values, df['station.latitude'].values)

error = pred - df[metric].values
print(df[metric].values)
print(pred)
print(error)
print(np.sqrt(np.mean(error ** 2)))

gp3dmodel = GP3DModel()
gp3dmodel.train(df, np.log(df[metric].values) - np.log(pred))

model = ProductModel(irimodel, LogSpaceModel(gp3dmodel))
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
plt.draw_title(metric + ' iri')
plt.write('/output/%s-iri.svg' % (metric))
plt.write('/output/%s-iri.png' % (metric))

plt2 = Plot(metric, dt.datetime.now(timezone.utc))
plt.scale_generic()
plt.draw_contour(gp3dmodel)
plt.draw_title(metric + ' residual')
plt.write('/output/%s-residual.svg' % (metric))
plt.write('/output/%s-residual.png' % (metric))

