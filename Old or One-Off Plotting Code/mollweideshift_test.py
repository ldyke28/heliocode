import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
import scipy.interpolate
import scipy.integrate
import scipy.stats
import cartopy as cr
import iris

"""# Target case
pole_lat = 90-5.3
pole_lon = 0
cent_lon = 0
rotated_pole2 = cr.crs.RotatedPole( pole_latitude = pole_lat, 
                                pole_longitude = pole_lon,
                                central_rotated_longitude = cent_lon)
# Create plot figure and axes
ax = plt.axes(projection=cr.crs.Mollweide())
ax.set_global()

# Plot the graticule
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-180,180,30), 
             ylocs=range(-90,90,30)) #draw_labels=True NOT allowed

# Plot some texts at various locations
lonlats = [ [0,0, '(0,0)'], [0,45, '(0,45)'], [0,90, '(0,90)'],
            [-105,0, '(-105,0)'], [-150,0, '(-150,0)'], [60,0, '(60,0)'], [90,0, '(90,0)']]
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=8, fontweight='ultralight', color="k", transform=rotated_pole2)

plt.show()"""

# load some sample iris data
fname = iris.sample_data_path('rotated_pole.nc')
temperature = iris.load_cube(fname)

# iris comes complete with a method to put bounds on a simple point
# coordinate. This is very useful...
temperature.coord('grid_latitude').guess_bounds()
temperature.coord('grid_longitude').guess_bounds()

# turn the iris Cube data structure into numpy arrays
gridlons = temperature.coord('grid_longitude').contiguous_bounds()
gridlats = temperature.coord('grid_latitude').contiguous_bounds()
temperature = temperature.data

# set up a map
ax = plt.axes(projection=cr.crs.PlateCarree())

# define the coordinate system that the grid lons and grid lats are on
rotated_pole = cr.crs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
plt.pcolormesh(gridlons, gridlats, temperature, transform=rotated_pole)

ax.coastlines()

plt.show()

