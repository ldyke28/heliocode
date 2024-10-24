import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
import scipy.interpolate

file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/maxwellian3d_testdata.txt", delimiter=',')

nH = 0.195 # hydrogen density in num/cm^3
tempH = 7500 # LISM hydrogen temperature in K
mH = 1.6736*10**(-27) # mass of hydrogen in kg
vthn = np.sqrt(2*1.381*10**(-29)*tempH/mH)

theta = 275
vsc = 30000 # velocity of spacecraft in m/s
thetarad = theta*np.pi/180 # expressing the value of theta in radians
# calculating the shift of the particle velocities into the spacecraft frame
xshiftfactor = -vsc*np.cos(thetarad + np.pi/2)
yshiftfactor = -vsc*np.sin(thetarad + np.pi/2)

#vxarray = np.arange(-45000, 45000, 2000)
#vyarray = np.arange(-45000, 45000, 2000)
#vzarray = np.arange(-45000, 45000, 2000)

#vxkms = vxarray/1000
#vykms = vyarray/1000
#vzkms = vzarray/1000

vxstore = np.array([])
vystore = np.array([])
vzstore = np.array([])
#fstore = np.array([])
particleflux = np.array([])

for i in range(np.shape(file)[0]):
    vxstore = np.append(vxstore, file[i,0]*1000)
    vystore = np.append(vystore, file[i,1]*1000)
    vzstore = np.append(vzstore, file[i,2]*1000)
    particleflux = np.append(particleflux, file[i,3])

vxkms = vxstore/1000
vykms = vystore/1000
vzkms = vzstore/1000

"""for i in tqdm(range(vxarray.size)):
    for j in tqdm(range(vyarray.size)):
        for k in range(vzarray.size):
            vxstore = np.append(vxstore, [vxarray[i]])
            vystore = np.append(vystore, [vyarray[j]])
            vzstore = np.append(vzstore, [vzarray[k]])
            # calculating PSD values based on a Maxwellian centered on the ISM flow speed
            fstore = np.append(fstore, nH*(1/(np.sqrt(np.pi)*vthn))**3*np.exp(-((vxkms[i]+26)**2 + (vykms[j])**2 + (vzkms[k])**2)/(10.195)**2))
            #fstore = np.append(fstore, [np.exp(-((vxkms[i]+26)**2 + (vykms[j])**2 + (vzkms[k])**2)/((10.195)**2))])"""

# shifting the velocities to calculate flux in the spacecraft frame
vxshift = vxstore + xshiftfactor
vyshift = vystore + yshiftfactor

vsquaredshift =  vxshift**2 + vyshift**2 

#particleflux = (vsquaredshift/1000)/(mH*6.242*10**(16)) * fstore # calculating particle flux at the device (https://link.springer.com/chapter/10.1007/978-3-030-82167-8_3 chapter 3.3)
# converting to cm^-2 s^-1 ster^-1 keV^-1

# finding unique vz values and ordering them as they are in the original file
vzshape = 0
newvz = np.array([])
for i in range(vxstore.size):
    newvz = np.append(newvz, [vzstore[i]])
    if vystore[i+1] != vystore[i]:
        vzshape = i+1
        break

#print(newvz)
#print(vzshape)

# same with vy
vyshape = 0
newvy = np.array([])
for i in range(int(vxstore.size/vzshape)):
    if vxstore[i*vzshape-1] != vxstore[i*vzshape] and i != 0:
        vyshape = i
        break
    newvy = np.append(newvy, [vystore[i*vzshape]])
    
#print(newvy)
#print(vyshape)

# same with vz
vxshape = int(vxstore.size/vyshape/vzshape)
newvx = np.array([])
for i in range(vxshape):
    newvx = np.append(newvx, [vxstore[vyshape*vzshape*i]])

#print(newvx)

# reshaping the array of flux values to be three-dimensional on the grid of vx, vy, and vz final conditions
pfreshape = np.zeros((vxshape, vyshape, vzshape))
for i in range(vxshape):
    for j in range(vyshape):
        for k in range(vzshape):
            pfreshape[i][j][k] = particleflux[vyshape*vzshape*i + vzshape*j + k]


vxgrid, vygrid, vzgrid = np.meshgrid(newvx, newvy, newvz, indexing='ij')
# interpolating the values of the PSD/VDF on the grid from the simulation
interpvdf = scipy.interpolate.RegularGridInterpolator((newvx, newvy, newvz), pfreshape, bounds_error=False, fill_value=None)

print(interpvdf([-31000,-4000, 5000]))

testphi = np.linspace(0, 2*np.pi, 200)
testtheta = np.linspace(-np.pi/2, np.pi/2, 100)

testvx = np.array([])
testvy = np.array([])
testvz = np.array([])
testpf = np.array([])
#testvmag = np.array([10000, 20000, 30000, 40000])
testvmag = np.arange(35000, 45000, 2000)
#testvmag = np.array([10000])

for i in tqdm(range(testphi.size)):
    for j in range(testtheta.size):
        for k in range(testvmag.size):
            # calculating vx, vy, vz values while shifting the center into the spacecraft frame
            currentvx = testvmag[k]*np.cos(testphi[i])*np.cos(testtheta[j]) + xshiftfactor
            currentvy = testvmag[k]*np.sin(testphi[i])*np.cos(testtheta[j]) + yshiftfactor
            currentvz = testvmag[k]*np.sin(testtheta[j])
            testvx = np.append(testvx, currentvx)
            testvy = np.append(testvy, currentvy)
            testvz = np.append(testvz, currentvz)
            testpf = np.append(testpf, interpvdf([currentvx, currentvy, currentvz]))


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


"""fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
scatterplot = ax3d.scatter3D(vxstore[:]/1000, vystore[:]/1000, vzstore[:]/1000, c=particleflux[:], cmap='rainbow', s=.1, norm=matplotlib.colors.LogNorm(vmin=10**(-11)))
#scatterplot = ax3d.scatter3D(vxstore[:]/1000, vystore[:]/1000, vzstore[:]/1000, c=particleflux[:], cmap='rainbow', s=.1)
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
plt.show()"""

vxkms2 = vxstore/1000
vykms2 = vystore/1000
vzkms2 = vzstore/1000

for i in range(vxstore.size):
    # normalizing the velocity components so the velocities all have a magnitude of 1 km/s
    vmag = np.sqrt(vxkms2[i]**2 + vykms2[i]**2 + vzkms2[i]**2)
    vxkms2[i] = vxkms2[i]/vmag
    vykms2[i] = vykms2[i]/vmag
    vzkms2[i] = vzkms2[i]/vmag

    vxstore[i] = vxkms2[i]*1000
    vystore[i] = vykms2[i]*1000
    vzstore[i] = vzkms2[i]*1000

"""fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
scatterplot = ax3d.scatter3D(vxstore[:]/1000, vystore[:]/1000, vzstore[:]/1000, c=particleflux[:], cmap='rainbow', s=.1, norm=matplotlib.colors.LogNorm(vmin=10**(-11)))
#scatterplot = ax3d.scatter3D(vxstore[:]/1000, vystore[:]/1000, vzstore[:]/1000, c=particleflux[:], cmap='rainbow', s=.1)
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
plt.show()"""

#vxkms2 = testvx/1000
#vykms2 = testvy/1000
#vzkms2 = testvz/1000

testvx = testvx/1000
testvy = testvy/1000
testvz = testvz/1000

# initializing arrays to store the azimuthal/polar angles
phi = np.zeros(testvx.size)
theta = np.zeros(testvx.size)

# calculating the magnitude of the velocities in the vx-vy plane
radxy = np.sqrt(testvx**2 + testvy**2)

#flipping the signs for all velocities to get the directions from which they are seen as coming in
#vxkms2 = -vxkms2
#vykms2 = -vykms2
#vzkms2 = -vzkms2
testvx = -testvx
testvy = -testvy
testvz = -testvz

vmags = np.sqrt(testvx**2 + testvy**2 + testvz**2)

for i in range(testvx.size):
    # calculating the azimuthal and polar angles for each point
    phi[i] = np.arccos(testvx[i]/radxy[i])
    if testvy[i] < 0:
        phi[i] = -phi[i]
    theta[i] = np.arcsin(testvz[i]/vmags[i])

# defining the bin width in degrees and radians
ibexvaw = 6
ibexvawr = ibexvaw*np.pi/180

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
            continue

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

im = ax.pcolormesh(Lon,Lat,psdtracker, cmap='rainbow', norm=matplotlib.colors.LogNorm())
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
#plt.xlabel("$\phi$")
#plt.ylabel("theta")
fig.colorbar(im, ax=ax)
plt.show()