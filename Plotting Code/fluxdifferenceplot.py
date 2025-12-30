import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
import scipy.interpolate
import scipy.integrate
import scipy.stats
import cartopy.crs as ccrs


#############################################################################################################
# RELEVANT VARIABLES/PARAMETERS
#############################################################################################################

# one year divided by 60 (6 degree shift for IBEX) - 525909.09090833333 s

nH = 0.195 # hydrogen density in num/cm^3
tempH = 7500 # LISM hydrogen temperature in K
mH = 1.6736*10**(-27) # mass of hydrogen in kg
vthn = np.sqrt(2*1.381*10**(-29)*tempH/mH) # thermal velocity of LISM H

theta = 275 # angle with respect to the upwind axis of the target point
vsc = 30000 # velocity of spacecraft in m/s
thetarad = theta*np.pi/180 # expressing the value of theta in radians
# calculating the shift of the particle velocities into the spacecraft frame
xshiftfactor = -vsc*np.cos(thetarad + np.pi/2)
yshiftfactor = -vsc*np.sin(thetarad + np.pi/2)

# defining the spacecraft bin width in degrees and radians
ibexvaw = 6
ibexvawr = ibexvaw*np.pi/180
ibexvahwr = ibexvawr/2

# set of velocity magnitudes to make shells of points
# CHANGE THIS FOR DIFFERENT TARGET POINT FOR BETTER ACCURACY
testvmag = np.arange(25000, 100000, 5000)

# deciding whether the flux should be calculated in the spacecraft frame (True) or the inertial frame (False)
shiftflux = True
# deciding if the view of the Mollweide should be in the spacecraft frame (True) or the inertial frame (False)
shiftorigin = True

esas = np.array([.010, .01944, .03747, .07283]) # value for ESA1's high energy boundary in keV

def eVtov(esaenergy):
    # converts energy in eV to velocity in m/s
    return np.sqrt(esaenergy*1.602*10**(-19)/(.5 * 1.6736*10**(-27)))


#############################################################################################################

file1 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/-17pi36_5p5yr_lya_Federicodist_datamu_fixed_3500vres_post.txt", delimiter=',')
file2 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/-17pi36_5p5yr_lya_analyticbc_datamu_order5_newpi_3500vres_post.txt", delimiter=',')

phi1 = np.array([])
theta1 = np.array([])
flux1 = np.array([])
vmag1 = np.array([])
phi2 = np.array([])
theta2 = np.array([])
flux2 = np.array([])
vmag2 = np.array([])

# unpacking data from both files
for i in range(np.shape(file1)[0]):
    phi1 = np.append(phi1, file1[i,0])
    theta1 = np.append(theta1, file1[i,1])
    flux1 = np.append(flux1, file1[i,2])
    vmag1 = np.append(vmag1, file1[i,3])

for i in range(np.shape(file2)[0]):
    phi2 = np.append(phi2, file2[i,0])
    theta2 = np.append(theta2, file2[i,1])
    flux2 = np.append(flux2, file2[i,2])
    vmag2 = np.append(vmag2, file2[i,3])


# creating grid points in angle space
phibounds = np.linspace(-np.pi, np.pi, int(360/ibexvaw+1))
thetabounds = np.linspace(-np.pi/2, np.pi/2, int(180/ibexvaw+1))

# creating an array to track the PSD value at the center of the cells made by the grid points
psdtracker1 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounter1 = np.zeros((phibounds.size-1, thetabounds.size-1))
psdtracker2 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounter2 = np.zeros((phibounds.size-1, thetabounds.size-1))


# finds which cell each velocity point lies in
for k in tqdm(range(phi1.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi1[k] < phibounds[i+1] and thetabounds[j] <= theta1[k] < thetabounds[j+1]:
                # adding the value of the PSD to the associated value for the cell and exiting the loop
                #psdtracker[i,j] += particleflux[k]
                psdtracker1[i,j] += flux1[k]
                bincounter1[i,j] += 1
                checker = True
        if checker == True:
            break

# finds which cell each velocity point lies in
for k in tqdm(range(phi2.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi2[k] < phibounds[i+1] and thetabounds[j] <= theta2[k] < thetabounds[j+1]:
                # adding the value of the PSD to the associated value for the cell and exiting the loop
                #psdtracker[i,j] += particleflux[k]
                psdtracker2[i,j] += flux2[k]
                bincounter2[i,j] += 1
                checker = True
        if checker == True:
            break

# dividing the total summed PSD in each bin by the number of points in that bin
psdtracker1 = psdtracker1/bincounter1
psdtracker2 = psdtracker2/bincounter2

psdtracker1 = np.transpose(psdtracker1)
psdtracker2 = np.transpose(psdtracker2)

psdtrackerdiff = psdtracker2 - psdtracker1

# defining a grid for the midpoints of the cells
adjphib = np.linspace(-180, 180, int(360/ibexvaw+1))
adjthetab = np.linspace(-90, 90, int(180/ibexvaw+1))

# making a grid from the above
Lon,Lat = np.meshgrid(adjphib,adjthetab)

pole_lat = 90-5.3
pole_lon = 0
cent_lon = 180
rotated_pole2 = ccrs.RotatedPole( pole_latitude = pole_lat, 
                                pole_longitude = pole_lon,
                                central_rotated_longitude = cent_lon)
eclipticzero = -105
lonlats = [ [15,0, '-120$^{\circ}$'], [eclipticzero,45, '45$^{\circ}$'], [eclipticzero,90, '90$^{\circ}$'], [eclipticzero,-45, '-45$^{\circ}$'],
            [-15,0, '-90$^{\circ}$'], [eclipticzero,0, '0$^{\circ}$'], [165,0, '90$^{\circ}$'], [eclipticzero,15, '15$^{\circ}$'],
            [eclipticzero,30, '30$^{\circ}$'], [eclipticzero,75, '75$^{\circ}$'], [eclipticzero,60, '60$^{\circ}$'], [eclipticzero,-15, '-15$^{\circ}$'],
            [eclipticzero,-30, '-30$^{\circ}$'], [eclipticzero,-60, '-60$^{\circ}$'], [eclipticzero,-75, '-75$^{\circ}$'], [-75,0, '-30$^{\circ}$'], [-45,0, '-60$^{\circ}$'],
            [45,0, '-150$^{\circ}$'], [75,0, '180$^{\circ}$'], [105,0, '150$^{\circ}$'], [135,0, '120$^{\circ}$'], [-135,0, '30$^{\circ}$'], [-165,0, '60$^{\circ}$']]
fig = plt.figure(figsize=(9,6))
# Create plot figure and axes
ax = plt.axes(projection=ccrs.Mollweide())

# Plot the graticule
im = ax.pcolormesh(Lon,Lat,psdtrackerdiff, cmap='berlin', transform=rotated_pole2, vmin=-10**(6),vmax=10**(6))
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=8, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Difference in Differential Flux at Detector (cm$^-2$ s$^-1$ sr$^-1$ keV$^-1$)')
plt.show()


# arrays to track flux values and bin counts in each ESA range
pftrackeresa1 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa1 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa2 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa2 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa3 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa3 = np.zeros((phibounds.size-1, thetabounds.size-1))

pftrackeresa21 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa21 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa22 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa22 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa23 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa23 = np.zeros((phibounds.size-1, thetabounds.size-1))

# converting each ESA boundary to m/s to compare with velocities
esa1lowv = eVtov(esas[0]*1000)
esa12 = eVtov(esas[1]*1000)
esa23 = eVtov(esas[2]*1000)
esa3upb = eVtov(esas[3]*1000)

geometricfactor = 1.41*10**(-5) # geometric factor in cm^2 sr for IBEX-Lo, after https://iopscience.iop.org/article/10.3847/1538-4365/adcd5c/pdf

print(esa1lowv)
print(esa3upb)

vmags1kms = vmag1*1000
vmags2kms = vmag2*1000

# checking which bin and in which ESA range each point falls
for k in tqdm(range(phi1.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi1[k] < phibounds[i+1] and thetabounds[j] <= theta1[k] < thetabounds[j+1]:
                if esa1lowv < vmags1kms[k] < esa12:
                    pftrackeresa1[i,j] += flux1[k]
                    bincounteresa1[i,j] += 1
                elif esa12 < vmags1kms[k] < esa23:
                    pftrackeresa2[i,j] += flux1[k]
                    bincounteresa2[i,j] += 1
                elif esa23 < vmags1kms[k] < esa3upb:
                    pftrackeresa3[i,j] += flux1[k]
                    bincounteresa3[i,j] += 1
                checker = True
        if checker == True:
            break

# checking which bin and in which ESA range each point falls
for k in tqdm(range(phi2.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi2[k] < phibounds[i+1] and thetabounds[j] <= theta2[k] < thetabounds[j+1]:
                if esa1lowv < vmags2kms[k] < esa12:
                    pftrackeresa21[i,j] += flux2[k]
                    bincounteresa21[i,j] += 1
                elif esa12 < vmags2kms[k] < esa23:
                    pftrackeresa22[i,j] += flux2[k]
                    bincounteresa22[i,j] += 1
                elif esa23 < vmags2kms[k] < esa3upb:
                    pftrackeresa23[i,j] += flux2[k]
                    bincounteresa23[i,j] += 1
                checker = True
        if checker == True:
            break

# normalizing by bin count
pftrackeresa1 = pftrackeresa1/bincounteresa1
pftrackeresa2 = pftrackeresa2/bincounteresa2
pftrackeresa3 = pftrackeresa3/bincounteresa3

pftrackeresa21 = pftrackeresa21/bincounteresa21
pftrackeresa22 = pftrackeresa22/bincounteresa22
pftrackeresa23 = pftrackeresa23/bincounteresa23

pftrackeresa1 = np.transpose(pftrackeresa1) # transposing the PSD value array to work with the grid
pftrackeresa2 = np.transpose(pftrackeresa2)
pftrackeresa3 = np.transpose(pftrackeresa3)

pftrackeresa21 = np.transpose(pftrackeresa21) # transposing the PSD value array to work with the grid
pftrackeresa22 = np.transpose(pftrackeresa22)
pftrackeresa23 = np.transpose(pftrackeresa23)

pftrackeresa1diff = pftrackeresa21 - pftrackeresa1
pftrackeresa2diff = pftrackeresa22 - pftrackeresa2
pftrackeresa3diff = pftrackeresa23 - pftrackeresa3

pole_lat = 90-5.3
pole_lon = 0
cent_lon = 180
rotated_pole2 = ccrs.RotatedPole( pole_latitude = pole_lat, 
                                pole_longitude = pole_lon,
                                central_rotated_longitude = cent_lon)
eclipticzero = -105
lonlats = [ [15,0, '-120$^{\circ}$'], [eclipticzero,45, '45$^{\circ}$'], [eclipticzero,90, '90$^{\circ}$'], [eclipticzero,-45, '-45$^{\circ}$'],
            [-15,0, '-90$^{\circ}$'], [eclipticzero,0, '0$^{\circ}$'], [165,0, '90$^{\circ}$'], [eclipticzero,15, '15$^{\circ}$'],
            [eclipticzero,30, '30$^{\circ}$'], [eclipticzero,75, '75$^{\circ}$'], [eclipticzero,60, '60$^{\circ}$'], [eclipticzero,-15, '-15$^{\circ}$'],
            [eclipticzero,-30, '-30$^{\circ}$'], [eclipticzero,-60, '-60$^{\circ}$'], [eclipticzero,-75, '-75$^{\circ}$'], [-75,0, '-30$^{\circ}$'], [-45,0, '-60$^{\circ}$'],
            [45,0, '-150$^{\circ}$'], [75,0, '180$^{\circ}$'], [105,0, '150$^{\circ}$'], [135,0, '120$^{\circ}$'], [-135,0, '30$^{\circ}$'], [-165,0, '60$^{\circ}$']]
fig = plt.figure(figsize=(9,6))
# Create plot figure and axes
ax = plt.axes(projection=ccrs.Mollweide())

# Plot the graticule
im = ax.pcolormesh(Lon,Lat,pftrackeresa1diff*(esas[1]-esas[0]), cmap='berlin', transform=rotated_pole2, vmin=-10**(4),vmax=10**(4))
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=8, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Intensity at Detector (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.show()

fig = plt.figure(figsize=(9,6))
# Create plot figure and axes
ax = plt.axes(projection=ccrs.Mollweide())

# Plot the graticule
im = ax.pcolormesh(Lon,Lat,pftrackeresa2diff*(esas[2]-esas[1]), cmap='berlin', transform=rotated_pole2, vmin=-10**(4),vmax=10**(4))
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=8, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Intensity at Detector (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.show()

fig = plt.figure(figsize=(9,6))
# Create plot figure and axes
ax = plt.axes(projection=ccrs.Mollweide())

# Plot the graticule
im = ax.pcolormesh(Lon,Lat,pftrackeresa3diff*(esas[3]-esas[2]), cmap='berlin', transform=rotated_pole2, vmin=-10**(4),vmax=10**(4))
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=8, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Intensity at Detector (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.show()