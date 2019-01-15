import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from pandas.plotting import table
import datetime as dt
from datetime import timezone
import numpy as np
import datetime as dt
import cartopy.feature
from cartopy.feature.nightshade import Nightshade
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
matplotlib.style.use('ggplot')
from scipy import interpolate
import scipy
import os
import sys
import logging
import urllib.request, json
from pandas.io.json import json_normalize
import geojsoncontour

metric = sys.argv[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        #logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()
now = dt.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
date = dt.datetime.now(timezone.utc) #.strftime('%Y, %m, %d, %H, %M')

def sph_to_xyz(lon, lat):
    lon = lon * np.pi / 180.
    lat = lat * np.pi / 180.
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z

def main():
    SPH_ORDER = 3
    SPH_WEIGHT = 0.75
    RESIDUAL_WEIGHT = 1

    plt.clf()

    with urllib.request.urlopen(os.getenv("METRICS_URI")) as url:
        data = json.loads(url.read().decode())

    df = json_normalize(data)
  
    #delete low confidence measurements
    df = df.drop(df[pd.to_numeric(df.cs) == 0].index)
    df = df.drop(df[df[metric] == 0].index)
    df = df.dropna(subset=[metric])

    #filter out data older than 1hr
    age = (dt.datetime.now() - dt.timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M')
    df = df.loc[df['time'] > age]

    df['time'] = pd.to_datetime(df.time)
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')

    df[[metric]] = df[[metric]].apply(pd.to_numeric)
    df[['station.longitude']] = df[['station.longitude']].apply(pd.to_numeric)
    df[['station.latitude']] = df[['station.latitude']].apply(pd.to_numeric)
    df['longitude_radians'] = df['station.longitude'] * np.pi / 180.
    df['latitude_radians'] = (df['station.latitude'] + 90) * np.pi / 180.
    df[['cs']] = df[['cs']].apply(pd.to_numeric)

    df = df.dropna(subset=[metric])
    df.loc[df['station.longitude'] > 180, 'station.longitude'] = df['station.longitude'] - 360

    df.sort_values(by=['station.longitude'], inplace=True)

    sph = []
    for n in range(SPH_ORDER):
        for m in range(0-n,n+1):
            sph.append(scipy.special.sph_harm(m, n, df['longitude_radians'].values, df['latitude_radians'].values).reshape((-1,1)))
    sph = np.hstack(sph)
    print(sph)

    print(df[metric].values)
    coeff = scipy.linalg.lstsq(sph, df[metric].values)[0]
    print(coeff)

   
    numcols, numrows = 360, 160
    loni = np.linspace(-180, 180, numcols)
    lati = np.linspace(-80, 80, numrows)

    theta = loni * np.pi / 180.
    phi = (lati + 90) * np.pi / 180.

    zi = np.zeros((len(phi),len(theta)))
    theta, phi = np.meshgrid(theta, phi)

    df['pred'] = np.zeros(len(df))

    coeff_idx = 0
    for n in range(SPH_ORDER):
        for m in range(0-n,n+1):
            sh = scipy.special.sph_harm(m, n, theta, phi)
            print("sh:", sh)
            weight = 1 if n == 0 else SPH_WEIGHT
            zi = zi + weight * np.real(coeff[coeff_idx] * sh)
            df['pred'] = df['pred'] + weight * np.real(coeff[coeff_idx] * scipy.special.sph_harm(m, n, df['longitude_radians'].values, df['latitude_radians'].values))
            coeff_idx = coeff_idx + 1

    df['residual'] = df[metric] - df['pred']
    #plot data
    
    loni = np.linspace(-180, 180, numcols)
    lati = np.linspace(-80, 80, numrows)
    loni, lati = np.meshgrid(loni, lati)
    x, y, z = sph_to_xyz(df['station.longitude'].values, df['station.latitude'].values)
    t = df['residual'].values
    rbf = interpolate.Rbf(x, y, z, t, smooth=0.25)

    xxi, yyi, zzi = sph_to_xyz(loni, lati)
    resi = rbf(xxi, yyi, zzi)

    zi = zi + RESIDUAL_WEIGHT * resi
    
    fig = plt.figure(figsize=(16, 24))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    levels = 16
    contour_args = {}
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(1))

    if metric == 'mufd':
        levels = [3, 3.5, 4, 4.6, 5.3, 6.1, 7, 8.2, 9.5, 11, 12.6, 14.6, 16.9, 19.5, 22.6, 26, 30]
        contour_args['norm'] = matplotlib.colors.LogNorm(3.5,30, clip=False)

    mycontour = plt.contourf(loni, lati, zi, levels,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                alpha=0.3,
                **contour_args
                )
    
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '110m',
        edgecolor='face',
        facecolor=np.array((0xdd,0xdd,0xcc))/256.,
        zorder=-1
        )
        )
    ax.set_global()
    ax.add_feature(Nightshade(date, alpha=0.08))

    ax.grid(linewidth=.5, color='black', alpha=0.25, linestyle='--')
    ax.set_xticks([-180, -160, -140, -120,-100, -80, -60,-40,-20, 0, 20, 40, 60,80,100, 120,140, 160,180], crs=ccrs.PlateCarree())
    ax.set_yticks([-80, -60,-40,-20, 0, 20, 40, 60,80], crs=ccrs.PlateCarree())
    
    for index, row in df.iterrows():
      lon = float(row['station.longitude'])
      lat = float(row['station.latitude'])
      ax.text(lon, lat, int(row[metric]), fontsize=10,ha='left', transform=ccrs.PlateCarree()) 
    
#    plt.clabel(mycontour, inline=False, colors='black', fontsize=10, fmt='%.0f')

    CS2 = plt.contour(mycontour, linewidths=.5, alpha=0.66, levels=mycontour.levels[1::1])
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(mycontour, fraction=0.03, orientation='horizontal', pad=0.02, format=matplotlib.ticker.ScalarFormatter())
    #cbar.set_label('MHz') #TODO add unit
    cbar.add_lines(CS2)
    
    plt.title(metric + ' ' + str(now))
   
#    df = df[['station.name', 'time', metric, 'cs', 'altitude', 'station.longitude', 'station.latitude']]

#    df = df.round(2)

#    the_table = table(ax, df,
#          bbox=[0,-1.25,1,1],
#          cellLoc = 'left',)

#    for key, cell in the_table.get_celld().items():
#        cell.set_linewidth(.25)
 
    plt.tight_layout()
    plt.savefig('/output/{}.png'.format(metric), dpi=180,bbox_inches='tight')
    
    # Convert matplotlib contour to geojson
    """
    geojsoncontour.contourf_to_geojson(
        contourf=mycontour,
        geojson_filepath='/output/{}.geojson'.format(metric),
        min_angle_deg=3.0,
        ndigits=2,
        stroke_width=2,
        fill_opacity=0.5,
        )
    """


if __name__ == '__main__':
    main()
