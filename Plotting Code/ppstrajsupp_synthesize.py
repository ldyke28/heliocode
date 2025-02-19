import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
import scipy.interpolate

file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/3ddata/-17pi36_0yr_lya_Federicodist_datamu_fixed_3000vres.txt", delimiter=',')
suppfile = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/3ddata/lostpoints_-17pi36_0yr_lya_Federicodist_datamu_fixed_3000vres_revised.txt", delimiter=',')

vx1 = np.array([])
vy1 = np.array([])
vz1 = np.array([])
f1 = np.array([])
vx2 = np.array([])
vy2 = np.array([])
vz2 = np.array([])
f2 = np.array([])

# unpacking data from both files
for i in range(np.shape(file)[0]):
    vx1 = np.append(vx1, file[i,0])
    vy1 = np.append(vy1, file[i,1])
    vz1 = np.append(vz1, file[i,2])
    f1 = np.append(f1, file[i,3])

for i in range(np.shape(suppfile)[0]):
    vx2 = np.append(vx2, suppfile[i,0])
    vy2 = np.append(vy2, suppfile[i,1])
    vz2 = np.append(vz2, suppfile[i,2])
    f2 = np.append(f2, suppfile[i,3])

# lost points need to be added back after having been re-run
# this part of the code overwrites any 0 values assigned to lost points in the original file
for i in range(vx2.size):
    for j in range(vx1.size):
        if vx1[j] == vx2[i] and vy1[j] == vy2[i] and vz1[j] == vz2[i]:
            f1[j] = f2[i]

for i in range(f1.size):
    if f1[i] == -1:
        # points in Sun are set to -1 - this will not work for integration/interpolation, so we revert them to 0 here
        f1[i] = 0


#############################################################################################################
# RELEVANT VARIABLES/PARAMETERS
#############################################################################################################

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

# set of velocity magnitudes to make shells of points
testvmag = np.arange(30000, 90000, 5000)

vxlower = min(vx1)*1000
vxupper = max(vx1)*1000
vylower = min(vy1)*1000
vyupper = max(vy1)*1000
vzlower = min(vz1)*1000
vzupper = max(vz1)*1000

# deciding whether the flux should be calculated in the spacecraft frame (True) or the inertial frame (False)
shiftflux = True
# deciding if the view of the Mollweide should be in the spacecraft frame (True) or the inertial frame (False)
shiftorigin = True


#############################################################################################################

# initializing arrays to store velocity component/PSD/differential flux values
vxstore = np.array([])
vystore = np.array([])
vzstore = np.array([])
fstore = np.array([])
particleflux = np.array([])


for i in range(vx1.size):
    # adding the values from the files to arrays
    vxstore = np.append(vxstore, vx1[i]*1000)
    vystore = np.append(vystore, vy1[i]*1000)
    vzstore = np.append(vzstore, vz1[i]*1000)
    #particleflux = np.append(particleflux, file[i,3])
    fstore = np.append(fstore, f1[i])

# storing the velocity values in km/s
vxkms = vxstore/1000
vykms = vystore/1000
vzkms = vzstore/1000

# shifting the velocities to calculate flux in the spacecraft frame
vxshift = vxstore + xshiftfactor
vyshift = vystore + yshiftfactor

# calculating the velocity squared of the particles in the inertial frame
vsquared = vxstore**2 + vystore**2 + vzstore**2
# calculating the velocity squared of the particles after the shift into spacecraft frame
vsquaredshift =  vxshift**2 + vyshift**2 + vzstore**2

if shiftflux:
    particleflux = (vsquaredshift/1000)/(mH*6.242*10**(16)) * fstore # calculating particle flux at the device (https://link.springer.com/chapter/10.1007/978-3-030-82167-8_3 chapter 3.3)
else:
    particleflux = (vsquared/1000)/(mH*6.242*10**(16)) * fstore
# converting to cm^-2 s^-1 ster^-1 keV^-1

# finding unique vz values and ordering them as they are in the original file
vzshape = 0
newvz = np.array([])
for i in range(vxstore.size):
    newvz = np.append(newvz, [vzstore[i]])
    if vystore[i+1] != vystore[i]:
        vzshape = i+1
        break

# same with vy
vyshape = 0
newvy = np.array([])
for i in range(int(vxstore.size/vzshape)):
    if vxstore[i*vzshape-1] != vxstore[i*vzshape] and i != 0:
        vyshape = i
        break
    newvy = np.append(newvy, [vystore[i*vzshape]])

# same with vx
vxshape = int(vxstore.size/vyshape/vzshape)
newvx = np.array([])
for i in range(vxshape):
    newvx = np.append(newvx, [vxstore[vyshape*vzshape*i]])

# reshaping the array of flux values to be three-dimensional on the grid of vx, vy, and vz final conditions
pfreshape = np.zeros((vxshape, vyshape, vzshape))
for i in range(vxshape):
    for j in range(vyshape):
        for k in range(vzshape):
            pfreshape[i][j][k] = particleflux[vyshape*vzshape*i + vzshape*j + k]

# creating a grid using these new values
vxgrid, vygrid, vzgrid = np.meshgrid(newvx, newvy, newvz, indexing='ij')
# interpolating the values of the PSD/VDF on the grid from the simulation
interpvdf = scipy.interpolate.RegularGridInterpolator((newvx, newvy, newvz), pfreshape, bounds_error=False, fill_value=None)

#print(interpvdf([-31000,-4000, 5000]))

# setting an array of values for azimuthal/polar angles for sampling points
testphi = np.linspace(0, 2*np.pi, 200)
testtheta = np.linspace(-np.pi/2, np.pi/2, 100)

# initializing arrays to store values of points on shells used for processing
testvx = np.array([])
testvy = np.array([])
testvz = np.array([])
testpf = np.array([])

for i in tqdm(range(testphi.size)):
    for j in range(testtheta.size):
        for k in range(testvmag.size):
            # calculating vx, vy, vz values while shifting the center into the spacecraft frame
            # need to subtract the shift factor when moving the origin (whereas it's added to vectors to shift them)
            currentvx = testvmag[k]*np.cos(testphi[i])*np.cos(testtheta[j]) - xshiftfactor
            currentvy = testvmag[k]*np.sin(testphi[i])*np.cos(testtheta[j]) - yshiftfactor
            currentvz = testvmag[k]*np.sin(testtheta[j])
            testvx = np.append(testvx, currentvx)
            testvy = np.append(testvy, currentvy)
            testvz = np.append(testvz, currentvz)
            #if vxupper >= currentvx:
            if vxlower <= currentvx <= vxupper and vylower <= currentvy <= vyupper and vzlower <= currentvz <= vzupper:
                testpf = np.append(testpf, interpvdf([currentvx, currentvy, currentvz]))
            else:
                testpf = np.append(testpf, [0])

# plotting the set of these points in 3D
fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
scatterplot = ax3d.scatter3D(testvx[:]/1000, testvy[:]/1000, testvz[:]/1000, c=testpf[:], cmap='rainbow', s=.1, norm=matplotlib.colors.LogNorm(vmin=10**(-11)))
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
plt.show()

# converting the points to km/s
testvx = testvx/1000
testvy = testvy/1000
testvz = testvz/1000

if shiftorigin == True:
    # gives the option to shift the origin in velocity space when calculating angles
    # this allows us to view the mollweide from the inertial frame or the spacecraft frame
    testvx = (testvx + xshiftfactor/1000)
    testvy = (testvy + yshiftfactor/1000)

#testvxshifted = testvx
#testvyshifted = testvy

# initializing arrays to store the azimuthal/polar angles
phi = np.zeros(testvx.size)
theta = np.zeros(testvx.size)

# calculating the magnitude of the velocities in the vx-vy plane
radxy = np.sqrt(testvx**2 + testvy**2)

#flipping the signs for all velocities to get the directions from which they are seen as coming in
testvx = -testvx
testvy = -testvy
testvz = -testvz

# calculating velocity magnitudes for each point
vmags = np.sqrt(testvx**2 + testvy**2 + testvz**2)

for i in range(testvx.size):
    # calculating the azimuthal and polar angles for each point
    # need to include the shift because we're looking from the spacecraft frame
    phi[i] = np.arccos(testvx[i]/radxy[i])
    if testvy[i] < 0:
        phi[i] = -phi[i]
    theta[i] = np.arcsin(testvz[i]/vmags[i])

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
                psdtracker[i,j] += testpf[k]
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

im = ax.pcolormesh(Lon,Lat,psdtracker, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(4)))
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector')
plt.show()

lookdir1 = thetarad + np.pi/2
lookdir2 = thetarad - np.pi/2

if lookdir1 > np.pi:
    lookdir1 = lookdir1 - 2*np.pi
if lookdir2 < -np.pi:
    lookdir2 = lookdir2 + 2*np.pi

# at some point - center bin around look direction angle?

# need to transpose back to the original shape, then back again for the Mollweide
psdtracker = np.transpose(psdtracker)

for i in range(phibounds.size-1):
    for j in range(thetabounds.size-1):
        if not (phibounds[i] <= lookdir1 <= phibounds[i+1] or phibounds[i] <= lookdir2 <= phibounds[i+1]):
            # if the look direction line does not lie in the bin, zero the value
            psdtracker[i,j] = 0

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

im = ax.pcolormesh(Lon,Lat,psdtracker, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(4)))
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector')
plt.show()
