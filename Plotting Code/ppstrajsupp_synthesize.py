import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
import scipy.interpolate
import scipy.integrate
import scipy.stats

#####################################################################################################################################
# CODE INTRODUCTION
#####################################################################################################################################

"""
This code is intended to use the 3D data from the cluster that did not throw the odeint error as well as the data obtained from
re-calculating any points lost to this error to create an interpolated grid of PSD data within the simulation region. This data is
used to make an all-sky map of the differential flux for the given dataset. Here are the processes the code completes:

1. The data from both files is unpacked, and any values of (vx, vy, vz) for which the original file has a 0 and the lost points file
has a value sees the 0 in the array of the data from the original file be replaced by the corresponding lost points file data
2. Variables are established - THE TARGET POINT ANGLE MUST BE INPUT MANUALLY - and there is an option to shift the flux calculation/
the Mollweide view into the spacecraft frame
3. The differential flux is calculated from the PSD values, and the grid is reshaped so the data can be interpolated across the
simulation region
4. Centering around the origin in the spacecraft frame, shells of points at various velocity magnitudes are given in phase space,
equally spaced angularly. These points have their value of the flux calculated if they are within the original simulation bounds, but
are otherwise set to 0
5. Those points are plotted
6. The corresponding look angle relative to the spacecraft is calculated for each of the points, such that the direction from which
they came can be determined
7. The points are all sorted into bins the width of the IBEX viewing angle, with their flux value averaged in each bin
8. The points are plotted on the Mollweide projection, to mimic how IBEX plots these values
9. Another version of the plot is constructed by only considering points that are within the IBEX viewing angle of the current IBEX
look direction, to further mimic what iBEX could observe
"""

#####################################################################################################################################

fname = "293deg_ibexshift_lya_Federicodist_datamu_order5_22yrsearlier_3500vres"

file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/" + fname + ".txt", delimiter=',')
suppfile = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/lostpoints_" + fname + "_revised.txt", delimiter=',')

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

esas = np.array([.010, .01944, .03747, .07283]) # value for ESA1's high energy boundary in keV

def eVtov(esaenergy):
    # converts energy in eV to velocity in m/s
    return np.sqrt(esaenergy*1.602*10**(-19)/(.5 * 1.6736*10**(-27)))


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
    particleflux = (vsquaredshift/(10**6))/(mH*6.242*10**(16)) * fstore # calculating particle flux at the device (https://link.springer.com/chapter/10.1007/978-3-030-82167-8_3 chapter 3.3)
else:
    particleflux = (vsquared/(10**6))/(mH*6.242*10**(16)) * fstore
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

# doing the same with the PSD values
psdreshape = np.zeros((vxshape, vyshape, vzshape))
for i in range(vxshape):
    for j in range(vyshape):
        for k in range(vzshape):
            psdreshape[i][j][k] = fstore[vyshape*vzshape*i + vzshape*j + k]

# creating a grid using these new values
vxgrid, vygrid, vzgrid = np.meshgrid(newvx, newvy, newvz, indexing='ij')
# interpolating the values of the particle flux on the grid from the simulation
interpvdf = scipy.interpolate.RegularGridInterpolator((newvx, newvy, newvz), pfreshape, bounds_error=False, fill_value=None)
# interpolating the values of the PSD on the grid from the simulation
interpvdfpsds = scipy.interpolate.RegularGridInterpolator((newvx, newvy, newvz), psdreshape, bounds_error=False, fill_value=None)

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

# storing the values in arrays to be interpolated later
integralvx = newvx
integralvy = newvy
integralvz = newvz

# plotting the set of these points in 3D
fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
scatterplot = ax3d.scatter3D(testvx[:]/1000, testvy[:]/1000, testvz[:]/1000, c=testpf[:], cmap='rainbow', s=.1, norm=matplotlib.colors.LogNorm(vmin=10**(-8)))
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
cb.set_label('Differential Flux at Detector (cm$^-2$ s$^-1$ sr$^-1$ keV$^-1$)')
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

# calculating the upper and lower bounds for the IBEX viewing angle in each direction
lookdir1 = thetarad + np.pi/2
lookdir2 = thetarad - np.pi/2

lookdir1up = lookdir1 + ibexvawr/2
lookdir1low = lookdir1 - ibexvawr/2
lookdir2up = lookdir2 + ibexvawr/2
lookdir2low = lookdir2 - ibexvawr/2

if lookdir1up > np.pi:
    lookdir1up = lookdir1up - 2*np.pi
if lookdir1low > np.pi:
    lookdir1low = lookdir1low - 2*np.pi
if lookdir2up < -np.pi:
    lookdir2up = lookdir2up + 2*np.pi
if lookdir2low < -np.pi:
    lookdir2low = lookdir2low + 2*np.pi

if lookdir2up > np.pi:
    lookdir2up = lookdir2up - 2*np.pi
if lookdir2low > np.pi:
    lookdir2low = lookdir2low - 2*np.pi
if lookdir1up < -np.pi:
    lookdir1up = lookdir1up + 2*np.pi
if lookdir1low < -np.pi:
    lookdir1low = lookdir1low + 2*np.pi

# creating arrays to plot boundary lines at the viewing angle limits
ld1lowmarkers = lookdir1low*np.ones((thetabounds.size))
ld1highmarkers = lookdir1up*np.ones((thetabounds.size))
ld2lowmarkers = lookdir2low*np.ones((thetabounds.size))
ld2highmarkers = lookdir2up*np.ones((thetabounds.size))
thetatracker = np.zeros(theta.size)
for i in range(theta.size):
    thetatracker[i] = theta[i]

im = ax.pcolormesh(Lon,Lat,psdtracker, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(1),vmax=10**(6)))
#im = ax.pcolormesh(Lon,Lat,psdtracker, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-15)))
ax.plot(ld1lowmarkers, thetabounds, color='black', alpha=0.5)
ax.plot(ld1highmarkers, thetabounds, color='black', alpha=0.5)
ax.plot(ld2lowmarkers, thetabounds, color='black', alpha=0.5)
ax.plot(ld2highmarkers, thetabounds, color='black', alpha=0.5)
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector (cm$^-2$ s$^-1$ sr$^-1$ keV$^-1$)')
plt.show()



# at some point - center bin around look direction angle?

# new array to track PSD when only considering trajectories within the viewing angle of the look directions
psdtrackerld = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounterld = np.zeros((phibounds.size-1, thetabounds.size-1))

phitrackerld = np.array([])
thetatrackerld = np.array([])
psdfiletrackerld = np.array([])


for k in tqdm(range(phi.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi[k] < phibounds[i+1] and thetabounds[j] <= theta[k] < thetabounds[j+1]:
                if (lookdir1low < phi[k] < lookdir1up or lookdir2low < phi[k] < lookdir2up):
                    # only adds the PSD value if the point is within half viewing angle of the look direction
                    psdtrackerld[i,j] += testpf[k]
                    bincounterld[i,j] += 1
                    phitrackerld = np.append(phitrackerld, phi[k])
                    thetatrackerld = np.append(thetatrackerld, theta[k])
                    psdfiletrackerld = np.append(psdfiletrackerld, testpf[k])
                checker = True
        if checker == True:
            break

# normalizing by bin count
psdtrackerld = psdtrackerld/bincounterld

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)

# defining a grid for the midpoints of the cells
adjphib = np.linspace(-np.pi+ibexvawr, np.pi-ibexvawr, int(360/ibexvaw))
adjthetab = np.linspace(-np.pi/2+ibexvawr, np.pi/2-ibexvawr, int(180/ibexvaw))

# making a grid from the above
Lon,Lat = np.meshgrid(adjphib,adjthetab)

psdtrackerld = np.transpose(psdtrackerld) # transposing the PSD value array to work with the grid

# writing data to a file - need to change each time or it will overwrite previous file
file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/Newest3DData/" + fname + "_ibexview.txt", 'w')
for i in range(psdfiletrackerld.size):
    # writes vx, vy, and attenuated NPSD value
    file.write(str(phitrackerld[i]) + ',' + str(thetatrackerld[i]) + ',' + str(psdfiletrackerld[i]) + '\n')
file.close()

im = ax.pcolormesh(Lon,Lat,psdtrackerld, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(1),vmax=10**(6)))
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector (cm$^-2$ s$^-1$ sr$^-1$ keV$^-1$)')
plt.show()


#############################################################################################################################

# finding the slopes of the lines associated with the IBEX viewing angle
slope1 = np.tan(thetarad - np.pi/2 + ibexvahwr)
slope2 = np.tan(thetarad - np.pi/2 - ibexvahwr)

def viewangle1(x):
    #equation for the line of one boundary of the viewing angle
    return slope1*(x + xshiftfactor/1000) - yshiftfactor/1000

def viewangle2(x):
    # equation for the line of the other boundary of the viewing angle
    return slope2*(x + xshiftfactor/1000) - yshiftfactor/1000

vxrotate = np.zeros(integralvx.size)
vyrotate = np.zeros(integralvx.size)
rotangle = thetarad - np.pi/2 # angle tangent to spacecraft path
# applying rotation to put center of viewing angle on x-axis
for i in range(integralvx.size):
    vxrotate[i] = integralvx[i]*np.cos(-rotangle) - integralvy[i]*np.sin(-rotangle)
    vyrotate[i] = integralvx[i]*np.sin(-rotangle) + integralvy[i]*np.cos(-rotangle)

# applying shift to align origin with origin of spacecraft frame
vxrotate = vxrotate + vsc
# viewing angle slopes
upperslope = np.tan(ibexvahwr)
lowerslope = np.tan(-ibexvahwr)

def vaupper(x):
    # y = mx in the rotated and shifted frame for positive slope
    return upperslope*x

def valower(x):
    # same for negative slope
    return lowerslope*x

# ESA boundaries translated into km/s to use for integration
energybounds = eVtov(esas)/1000

def psdfindrotated(vyrotated, vxrotated, vz):
    # velocity component values in the rotated frame ascribed their PSD value in the original frame
    return interpvdf([(vxrotated-vsc)*np.cos(rotangle) - vyrotated*np.sin(rotangle), (vxrotated-vsc)*np.sin(rotangle) + vyrotated*np.cos(rotangle), vz])

def interpolatedvdfvalue(vy, vx, vz):
    # returns value of PSD at different values of vx and vy
    return interpvdf([vx, vy, vz])/nH

def velocitymoment(vy, vx, vz):
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    return vmag*interpvdf([vx, vy, vz])

def velocitymomentx(vy, vx, vz):
    return vx*interpvdf([vx, vy, vz])/nH

def velocitymomenty(vy, vx, vz):
    return vy*interpvdf([vx, vy, vz])/nH

# integrated PSD between zero and lower first bound
#psdintegrate1 = scipy.integrate.tplquad(psdfindrotated, 0, energybounds[0], valower, vaupper)
# integrated PSD between first two bounds (within first ESA)
#psdintegrate2 = scipy.integrate.tplquad(psdfindrotated, energybounds[0], energybounds[1], valower, vaupper)
print("at least I got here")
# integrated PSD across all of velocity space
xlowbound = min(integralvx)
xhighbound = max(integralvx)
ylowbound = min(integralvy)
yhighbound = max(integralvy)
zlowbound = min(integralvz)
zhighbound = max(integralvz)
print(xlowbound)
print(xhighbound)
print(ylowbound)
print(yhighbound)
print(zlowbound)
print(zhighbound)
buffer = 10
#psdintegrate3 = scipy.integrate.tplquad(interpolatedvdfvalue, xlowbound+buffer, xhighbound-buffer, ylowbound+buffer, yhighbound-buffer, zlowbound+buffer, zhighbound-buffer)
#psdintegratexs = scipy.integrate.tplquad(velocitymomentx, xlowbound+buffer, xhighbound-buffer, ylowbound+buffer, yhighbound-buffer, zlowbound+buffer, zhighbound-buffer)
#psdintegrateys = scipy.integrate.tplquad(velocitymomenty, xlowbound+buffer, xhighbound-buffer, ylowbound+buffer, yhighbound-buffer, zlowbound+buffer, zhighbound-buffer)
#psdintegratex = 1/psdintegrate3[0] * psdintegratexs[0]
#psdintegratey = 1/psdintegrate3[0] * psdintegrateys[0]

#print("Integrated psd is: " + str(psdintegrate1))
#print("Second integrated psd is: " + str(psdintegrate2))
#print("PSD Integrated across all velocity space: " + str(psdintegrate3))
#print("Velocity Moment Across Entire Space: (" + str(psdintegratex) + ", " + str(psdintegratey) + ")")

# generating vx, vy, vz values on a regular rectangular grid
vstep = 1000
vxreggrid = np.arange(xlowbound, xhighbound, vstep)
vyreggrid = np.arange(ylowbound, yhighbound, vstep)
vzreggrid = np.arange(zlowbound, zhighbound, vstep)

# initializing an array to hold values of PSD at each point on the grid
gridpsds = np.zeros((vxreggrid.size, vyreggrid.size, vzreggrid.size))
# initializing similar arrays to hold values of the velocity components times the PSD value
gridvxpsd = np.zeros((vxreggrid.size, vyreggrid.size, vzreggrid.size))
gridvypsd = np.zeros((vxreggrid.size, vyreggrid.size, vzreggrid.size))
gridvzpsd = np.zeros((vxreggrid.size, vyreggrid.size, vzreggrid.size))
for i in range(vxreggrid.size):
    for j in range(vyreggrid.size):
        for k in range(vzreggrid.size):
            currentpsd = interpvdfpsds([vxreggrid[i], vyreggrid[j], vzreggrid[k]])
            if currentpsd < 0:
                currentpsd = 0
            gridpsds[i,j,k] = currentpsd
            gridvxpsd[i,j,k] = vxreggrid[i]*currentpsd
            gridvypsd[i,j,k] = vyreggrid[j]*currentpsd
            gridvzpsd[i,j,k] = vzreggrid[k]*currentpsd

density = np.sum(gridpsds)
densityscaled = density # PSD's are given in cm^-3 km^-3 s^3
ux = np.sum(gridvxpsd)/density
uy = np.sum(gridvypsd)/density
uz = np.sum(gridvzpsd)/density

print("Density acquired by summation method is: " + str(densityscaled) + " 1/cm^3.")
print("Bulk velocity acquired by summation method is: (" + str(ux) + ", " + str(uy) + ", " + str(uz) + ") m/s.")

#############################################################################################################################

# calculating the count rates in each bin, plotting the corresponding Mollweides

# arrays to track flux values and bin counts in each ESA range
pftrackeresa1 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa1 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa2 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa2 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa3 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa3 = np.zeros((phibounds.size-1, thetabounds.size-1))

# converting each ESA boundary to m/s to compare with velocities
esa1lowv = eVtov(esas[0]*1000)
esa12 = eVtov(esas[1]*1000)
esa23 = eVtov(esas[2]*1000)
esa3upb = eVtov(esas[3]*1000)

geometricfactor = 1.41*10**(-5) # geometric factor in cm^2 sr for IBEX-Lo, after https://iopscience.iop.org/article/10.3847/1538-4365/adcd5c/pdf

print(esa1lowv)
print(esa3upb)
print(min(vmags))
print(max(vmags))

vmagskms = vmags*1000

# checking which bin and in which ESA range each point falls
for k in tqdm(range(phi.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi[k] < phibounds[i+1] and thetabounds[j] <= theta[k] < thetabounds[j+1]:
                if esa1lowv < vmagskms[k] < esa12:
                    pftrackeresa1[i,j] += testpf[k]
                    bincounteresa1[i,j] += 1
                elif esa12 < vmagskms[k] < esa23:
                    pftrackeresa2[i,j] += testpf[k]
                    bincounteresa2[i,j] += 1
                elif esa23 < vmagskms[k] < esa3upb:
                    pftrackeresa3[i,j] += testpf[k]
                    bincounteresa3[i,j] += 1
                checker = True
        if checker == True:
            break

# normalizing by bin count
pftrackeresa1 = pftrackeresa1/bincounteresa1
pftrackeresa2 = pftrackeresa2/bincounteresa2
pftrackeresa3 = pftrackeresa3/bincounteresa3

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)

# defining a grid for the midpoints of the cells
adjphib = np.linspace(-np.pi+ibexvawr, np.pi-ibexvawr, int(360/ibexvaw))
adjthetab = np.linspace(-np.pi/2+ibexvawr, np.pi/2-ibexvawr, int(180/ibexvaw))

# making a grid from the above
Lon,Lat = np.meshgrid(adjphib,adjthetab)

pftrackeresa1 = np.transpose(pftrackeresa1) # transposing the PSD value array to work with the grid
pftrackeresa2 = np.transpose(pftrackeresa2)
pftrackeresa3 = np.transpose(pftrackeresa3)

im = ax.pcolormesh(Lon,Lat,pftrackeresa1, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(1),vmax=10**(7)))
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector (cm$^-2$ s$^-1$ sr$^-1$ keV$^-1$)')
plt.show()

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)
im = ax.pcolormesh(Lon,Lat,pftrackeresa2, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(1),vmax=10**(7)))
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector (cm$^-2$ s$^-1$ sr$^-1$ keV$^-1$)')
plt.show()

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)
im = ax.pcolormesh(Lon,Lat,pftrackeresa3, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(1),vmax=10**(7)))
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector (cm$^-2$ s$^-1$ sr$^-1$ keV$^-1$)')
plt.show()

# converting each from particle flux to count rate

crtrackeresa1 = pftrackeresa1*geometricfactor*(esas[1]-esas[0])
crtrackeresa2 = pftrackeresa2*geometricfactor*(esas[2]-esas[1])
crtrackeresa3 = pftrackeresa3*geometricfactor*(esas[3]-esas[2])

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)
im = ax.pcolormesh(Lon,Lat,crtrackeresa1, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-5),vmax=10**(1)))
#im = ax.pcolormesh(Lon,Lat,crtrackeresa1, cmap='rainbow', norm=matplotlib.colors.LogNorm())
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Average Count Rates at Detector')
plt.show()

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)
im = ax.pcolormesh(Lon,Lat,crtrackeresa2, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-5),vmax=10**(1)))
#im = ax.pcolormesh(Lon,Lat,crtrackeresa2, cmap='rainbow', norm=matplotlib.colors.LogNorm())
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Average Count Rates at Detector')
plt.show()

fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)
im = ax.pcolormesh(Lon,Lat,crtrackeresa3, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-5),vmax=10**(1)))
#im = ax.pcolormesh(Lon,Lat,crtrackeresa3, cmap='rainbow', norm=matplotlib.colors.LogNorm())
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Average Count Rates at Detector')
plt.show()