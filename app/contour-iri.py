import data
import datetime as dt
from datetime import timezone
import sys
import numpy as np
from models import GP3DModel, IRISplineModel, HybridModel, LogSpaceModel, LinearModel, ProductModel, DifferenceModel
from plot import Plot
import statsmodels.api as sm
from statsmodels.tools import add_constant

metric = sys.argv[1]

df = data.get_data()
df = data.filter(df, 
        max_age = dt.datetime.now()-dt.timedelta(minutes=60),
        required_metrics = [metric],
        min_confidence = 0.01
        )

irimodel = IRISplineModel("/iri.latest")
irimodel.train(metric)

pred = irimodel.predict(df['station.longitude'].values, df['station.latitude'].values)

error = pred - df[metric].values
print(df[metric].values)
print(pred)
print(error)
print(np.sqrt(np.sum(error ** 2) / np.sum(df.cs.values)), np.sum(error) / np.sum(df.cs.values))

irimodel_orig = irimodel

if metric in ['mufd', 'fof2']:
    wls_model = sm.WLS(df[metric].values - pred, add_constant(pred, prepend=False), df.cs.values)
    wls_fit = wls_model.fit()
    coeff = wls_fit.params
    coeff[0] = coeff[0] + 1
    print(coeff)

    irimodel = LinearModel(irimodel, coeff[0], coeff[1])
    pred = irimodel.predict(df['station.longitude'].values, df['station.latitude'].values)
    error = pred - df[metric].values
    print(df[metric].values)
    print(pred)
    print(error)
    print(np.sqrt(np.sum(error ** 2) / np.sum(df.cs.values)), np.sum(error) / np.sum(df.cs.values))

gp3dmodel = GP3DModel()
gp3dmodel.train(df, np.log(df[metric].values) - np.log(pred))

model = ProductModel(irimodel, LogSpaceModel(gp3dmodel))
pred = model.predict(df['station.longitude'].values, df['station.latitude'].values)
error = pred - df[metric].values
print(df[metric].values)
print(pred)
print(error)
print(np.sqrt(np.sum(error ** 2) / np.sum(df.cs.values)), np.sum(error) / np.sum(df.cs.values))

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

residual_model = DifferenceModel(model, irimodel_orig)
plt2 = Plot(metric, dt.datetime.now(timezone.utc))
plt.scale_generic()
plt.draw_contour(residual_model)
plt.draw_title(metric + ' residual')
plt.write('/output/%s-residual.svg' % (metric))
plt.write('/output/%s-residual.png' % (metric))

