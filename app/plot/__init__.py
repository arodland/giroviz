import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.feature.nightshade import Nightshade
matplotlib.style.use('ggplot')
import numpy as np

class Plot:
    def __init__(self, metric_name, date):
        self.metric_name = metric_name
        self.date = date

        self.fig = plt.figure(figsize=(16,24))
        self.ax = plt.axes(projection=ccrs.PlateCarree())
        self.ax.set_global()
        self.ax.grid(linewidth=.5, color='black', alpha=0.25, linestyle='--')
        self.ax.set_xticks([-180, -160, -140, -120,-100, -80, -60,-40,-20, 0, 20, 40, 60,80,100, 120,140, 160,180], crs=ccrs.PlateCarree())
        self.ax.set_yticks([-80, -60,-40,-20, 0, 20, 40, 60,80], crs=ccrs.PlateCarree())
        self.ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '110m',
            edgecolor='face',
            facecolor=np.array((0xdd,0xdd,0xcc))/256.,
            zorder=-1
            )
        )

        self.ax.add_feature(Nightshade(self.date, alpha=0.08))


    def scale_common(self):
        self.cmap = plt.cm.get_cmap('viridis')
        self.cmap.set_under(self.cmap(1e-5))
        self.cmap.set_over(self.cmap(1 - 1e-5))

    def scale_generic(self):
        self.scale_common()
        self.levels = 16
        self.norm = matplotlib.colors.Normalize(clip=False)

    def scale_mufd(self):
        self.scale_common()
        self.levels = [3, 3.5, 4, 4.6, 5.3, 6.1, 7, 8.2, 9.5, 11, 12.6, 14.6, 16.9, 19.5, 22.6, 26, 30]
        self.norm = matplotlib.colors.LogNorm(3.5, 30, clip=False)

    def draw_contour(self, model, lon_min=-180, lon_max=180, lon_steps=360, lat_min=-90, lat_max=90, lat_steps=180):
        loni = np.linspace(lon_min, lon_max, lon_steps)
        lati = np.linspace(lat_min, lat_max, lat_steps)
        loni, lati = np.meshgrid(loni, lati)
        zi = model.predict(loni, lati)

        contour = plt.contourf(loni, lati, zi, self.levels,
                cmap=self.cmap,
                extend='both',
                transform=ccrs.PlateCarree(),
                alpha=0.3,
                norm=self.norm
                )
        CS2 = plt.contour(contour, linewidths=.5, alpha=.66, levels=contour.levels[1::1])

        prev = None
        levels = []
        for lev in CS2.levels:
            if prev is None or '%.0f'%(lev) != '%.0f'%(prev):
                levels.append(lev)
                prev = lev

        plt.clabel(CS2, levels, inline=True, fontsize=10, fmt='%.0f', use_clabeltext=True )
        cbar = plt.colorbar(contour, fraction=0.03, orientation='horizontal', pad=0.02, format=matplotlib.ticker.ScalarFormatter())
        cbar.add_lines(CS2)


    def draw_title(self, metric):
        plt.title(metric + ' ' + str(self.date.strftime('%Y-%m-%d %H:%M')))

    def draw_dots(self, df, metric):
        for index, row in df.iterrows():
            lon = float(row['station.longitude'])
            lat = float(row['station.latitude'])
            alpha = 0.2 + 0.6 * row.cs
            self.ax.text(lon, lat, int(row[metric] + 0.5),
                    fontsize=9,
                    ha='left',
                    transform=ccrs.PlateCarree(),
                    alpha=alpha,
                    bbox={
                        'boxstyle': 'circle',
                        'alpha': alpha - 0.1,
                        'color': self.cmap(self.norm(row[metric])),
                        'mutation_scale': 0.5
                        }
                    )

    def write(self, filename):
            plt.tight_layout()
            plt.savefig(filename, dpi=180, bbox_inches='tight')

