import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
import scipy.interpolate

#############################################################################################################
# RELEVANT VARIABLES/PARAMETERS
#############################################################################################################

# one year divided by 60 (6 degree shift for IBEX) - 525909.09090833333 s

nH = 0.195 # hydrogen density in num/cm^3
tempH = 7500 # LISM hydrogen temperature in K
mH = 1.6736*10**(-27) # mass of hydrogen in kg
vthn = np.sqrt(2*1.381*10**(-29)*tempH/mH) # thermal velocity of LISM H

theta = 293 # angle with respect to the upwind axis of the target point
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

file1 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/245deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file2 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/251deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file3 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/257deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file4 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/263deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file5 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/269deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file6 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/-17pi36_t0_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file7 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/281deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file8 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/287deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file9 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/293deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file10 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/299deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file11 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/305deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file12 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/239deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file13 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/311deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file14 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/233deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')
file15 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/227deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres_ibexview.txt", delimiter=',')



phi = np.array([])
theta = np.array([])
flux = np.array([])

# unpacking data from both files
for i in range(np.shape(file1)[0]):
    phi = np.append(phi, file1[i,0])
    theta = np.append(theta, file1[i,1])
    flux = np.append(flux, file1[i,2])

for i in range(np.shape(file2)[0]):
    phi = np.append(phi, file2[i,0])
    theta = np.append(theta, file2[i,1])
    flux = np.append(flux, file2[i,2])

for i in range(np.shape(file3)[0]):
    phi = np.append(phi, file3[i,0])
    theta = np.append(theta, file3[i,1])
    flux = np.append(flux, file3[i,2])

for i in range(np.shape(file4)[0]):
    phi = np.append(phi, file4[i,0])
    theta = np.append(theta, file4[i,1])
    flux = np.append(flux, file4[i,2])

for i in range(np.shape(file5)[0]):
    phi = np.append(phi, file5[i,0])
    theta = np.append(theta, file5[i,1])
    flux = np.append(flux, file5[i,2])

for i in range(np.shape(file6)[0]):
    phi = np.append(phi, file6[i,0])
    theta = np.append(theta, file6[i,1])
    flux = np.append(flux, file6[i,2])

for i in range(np.shape(file7)[0]):
    phi = np.append(phi, file7[i,0])
    theta = np.append(theta, file7[i,1])
    flux = np.append(flux, file7[i,2])

for i in range(np.shape(file8)[0]):
    phi = np.append(phi, file8[i,0])
    theta = np.append(theta, file8[i,1])
    flux = np.append(flux, file8[i,2])

for i in range(np.shape(file9)[0]):
    phi = np.append(phi, file9[i,0])
    theta = np.append(theta, file9[i,1])
    flux = np.append(flux, file9[i,2])

for i in range(np.shape(file10)[0]):
    phi = np.append(phi, file10[i,0])
    theta = np.append(theta, file10[i,1])
    flux = np.append(flux, file10[i,2])

for i in range(np.shape(file11)[0]):
    phi = np.append(phi, file11[i,0])
    theta = np.append(theta, file11[i,1])
    flux = np.append(flux, file11[i,2])

for i in range(np.shape(file12)[0]):
    phi = np.append(phi, file12[i,0])
    theta = np.append(theta, file12[i,1])
    flux = np.append(flux, file12[i,2])

for i in range(np.shape(file13)[0]):
    phi = np.append(phi, file13[i,0])
    theta = np.append(theta, file13[i,1])
    flux = np.append(flux, file13[i,2])

for i in range(np.shape(file14)[0]):
    phi = np.append(phi, file14[i,0])
    theta = np.append(theta, file14[i,1])
    flux = np.append(flux, file14[i,2])

for i in range(np.shape(file15)[0]):
    phi = np.append(phi, file15[i,0])
    theta = np.append(theta, file15[i,1])
    flux = np.append(flux, file15[i,2])

# creating grid points in angle space
phibounds = np.linspace(-np.pi, np.pi, int(360/ibexvaw+1))
thetabounds = np.linspace(-np.pi/2, np.pi/2, int(180/ibexvaw+1))

# creating an array to track the PSD value at the center of the cells made by the grid points
psdtracker = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounter = np.zeros((phibounds.size-1, thetabounds.size-1))



# finds which cell each velocity point lies in
for k in tqdm(range(phi.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi[k] < phibounds[i+1] and thetabounds[j] <= theta[k] < thetabounds[j+1]:
                # adding the value of the PSD to the associated value for the cell and exiting the loop
                #psdtracker[i,j] += particleflux[k]
                psdtracker[i,j] += flux[k]
                bincounter[i,j] += 1
                checker = True
        if checker == True:
            break

# dividing the total summed PSD in each bin by the number of points in that bin
psdtracker = psdtracker/bincounter

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)

# defining a grid for the midpoints of the cells
adjphib = np.linspace(-np.pi+ibexvawr, np.pi-ibexvawr, int(360/ibexvaw))
adjthetab = np.linspace(-np.pi/2+ibexvawr, np.pi/2-ibexvawr, int(180/ibexvaw))

# making a grid from the above
Lon,Lat = np.meshgrid(adjphib,adjthetab)

psdtracker = np.transpose(psdtracker) # transposing the PSD value array to work with the grid

im = ax.pcolormesh(Lon,Lat,psdtracker, cmap='rainbow')#, norm=matplotlib.colors.LogNorm(vmin=10**(1),vmax=10**(6)))
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector')
plt.show()