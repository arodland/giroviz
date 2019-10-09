import data
import datetime as dt
from datetime import timezone
import dateutil.parser
import sys
import numpy as np
from models import GP3DModel, IRISplineModel, HybridModel, LogSpaceModel, LinearModel, ProductModel, DifferenceModel
from plot import Plot
import statsmodels.api as sm
from statsmodels.tools import add_constant

metric = sys.argv[1]
nowtime = None
if len(sys.argv) > 2:
    nowtime = dateutil.parser.parse(sys.argv[2])
else:
    nowtime = dt.datetime.now(timezone.utc)

df = data.get_data()
df = data.filter(df, 
        max_age = dt.datetime.now()-dt.timedelta(minutes=60),
        required_metrics = [metric],
        min_confidence = 0.01
        )

irimodel = IRISplineModel("/iri.latest")
irimodel.train(metric)

irimodel_orig = irimodel

if len(df) == 0:
    model = irimodel
else:
    pred = irimodel.predict(df['station.longitude'].values, df['station.latitude'].values)
    error = pred - df[metric].values
    print(df[metric].values)
    print(pred)
    print(error)
    print(np.sqrt(np.sum(error ** 2) / np.sum(df.cs.values)), np.sum(error) / np.sum(df.cs.values))


    if metric in ['mufd', 'fof2']:
        wls_model = sm.WLS(df[metric].values - pred, add_constant(pred, prepend=False), df.cs.values)
        wls_fit = wls_model.fit_regularized(alpha=np.array([1,3]), L1_wt=0)
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

plt = Plot(metric, nowtime)
if metric == 'mufd':
    plt.scale_mufd()
elif metric == 'fof2':
    plt.scale_fof2()
else:
    plt.scale_generic()

plt.draw_contour(model)
plt.draw_dots(df, metric)
plt.draw_title(metric + ' iri')
plt.write('/output/%s-iri.svg' % (metric))
plt.write('/output/%s-iri.png' % (metric))

residual_model = DifferenceModel(model, irimodel_orig)
plt2 = Plot(metric, nowtime)
plt2.scale_generic()
plt2.draw_contour(residual_model)
plt2.draw_title(metric + ' residual')
plt2.write('/output/%s-residual.svg' % (metric))
plt2.write('/output/%s-residual.png' % (metric))

