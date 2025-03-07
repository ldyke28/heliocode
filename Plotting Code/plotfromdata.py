import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

vx = np.array([])
vy = np.array([])
f = np.array([])

filenum = 1

#file = open("C:\Users\lucas\Downloads\cosexprp_pi32_1p5e8_indirect_cosexppi.txt", "r")
file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/filterfulltdrp_0p5deg_-4yr_whole_newcx+pi_tclose300_r=1au_interpdist.txt", delimiter=',')
#file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Paper Data/cosexpminrp_17pi36_t0_indirect_cosexppi_tclose200_r=1au.txt", delimiter=',')
#file = np.loadtxt("C:/Users/lucas/Downloads/Data Files-20230406T214257Z-001/Data Files/lyaminrp_5pi6_0y_direct_cosexppi_tclose400_1.txt", delimiter=',')
#file = np.loadtxt("/Users/ldyke/Downloads/drive-download-20221019T183112Z-001/cosexprp_pi32_1p5e8_indirect_cosexppi.txt", delimiter=',')


#file2 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/cosexprp_5pi6_6p36675e9_indirect_cosexppi.txt", delimiter=',')
#file2 = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/cosexprp_5pi6_6p36675e9_indirect_cosexppi.txt", delimiter=',')

#file3 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/cosexprp_5pi6_6p36675e9_direct_cosexppi.txt", delimiter=',')


for i in range(np.shape(file)[0]):
    vx = np.append(vx, file[i,0])
    vy = np.append(vy, file[i,1])
    f = np.append(f, file[i,2])

if filenum == 2:
    for i in range(np.shape(file2)[0]):
        vx = np.append(vx, file2[i,0])
        vy = np.append(vy, file2[i,1])
        f = np.append(f, file2[i,2])

if filenum == 3:
    for i in range(np.shape(file2)[0]):
        vx = np.append(vx, file2[i,0])
        vy = np.append(vy, file2[i,1])
        f = np.append(f, file2[i,2])
    for i in range(np.shape(file3)[0]):
        vx = np.append(vx, file3[i,0])
        vy = np.append(vy, file3[i,1])
        f = np.append(f, file3[i,2])

#fig = plt.figure()
#fig.set_figwidth(10)
#fig.set_figheight(6)

#for i in range(f.size):
#    if f[i] < 10**(-11):
#        f[i] = 0

vdftopsd = False
# Parameters (density and temperature) taken from Federico's code
nH = 0.195 # hydrogen density in num/cm^3
tempH = 7500 # LISM hydrogen temperature in K
mH = 1.6736*10**(-27) # mass of hydrogen in kg
vthn = np.sqrt(2*1.381*10**(-29)*tempH/mH)
if vdftopsd:
    f = f*nH*(1/(np.sqrt(np.pi)*vthn))**3 # scaling for normalized PSD values to turn normalized VDF's into PSD's

fsize = 18
"""#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='rainbow', vmin=0, vmax=0.5221685831603383)
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='rainbow', vmin=0, vmax=0.0020312330211150137)
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='rainbow')
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-50), vmax=10**(-6)))
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-11)))
plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm())
plt.rcParams.update({'font.size': fsize})
cb = plt.colorbar()
#cb.set_label('Time at which orbit passes through 100 au (s)')
#cb.set_label('Travel Time from 100 au to Point of Interest (s)')
cb.set_label('Normalized Phase Space Density')
#plt.xlim([-50, 20])
#plt.ylim([-30, 45])
#plt.xlim([0, 25])
#plt.ylim([5, 60])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
#plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
#plt.suptitle('VDF at target, at t $\\approx$ ' + str(round(finalt/(oneyear), 3)) + ' years, drawn from Maxwellian at 100 au centered on $v_x$ = -26 km/s')
#plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
#plt.title('Target at (' + str(round(ibexpos[0]/au, 3)) + ' au, ' + str(round(ibexpos[1]/au, 3)) + ' au), Time Resolution Close to Target = ' + str(tstepclose) + ' s')
#plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
plt.show()"""



"""fig = plt.figure()
fig.set_figwidth(9)
fig.set_figheight(6)
levels = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0]
plt.tricontour(vx[:], vy[:], f[:], levels)
cb = plt.colorbar()
cb.set_label('f(r,v,t)')
plt.xlabel("vx at Target in km/s")
plt.ylabel("vy at Target in km/s")
plt.suptitle('Phase space population at target (t = 6.246e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
plt.title('Target at (-.707 au, .707 au)')
plt.show()"""

#######################################################################################################################################


for i in range(f.size):
    if f[i] == -1:
        # points in Sun are set to -1 - this will skew bulk velocities
        f[i] = 0

vxtimesf = np.sum(f*vx)
vytimesf = np.sum(f*vy)
bulkux = vxtimesf/np.sum(f)
bulkuy = vytimesf/np.sum(f)

print("The bulk velocity for this phase space plot is approximately (" + str(bulkux) + ", " + str(bulkuy) + ")")



#######################################################################################################################################

# section of code to calculate which trajectories could be observed by spacecraft - considers velocity shifts and viewing angle
vx = vx*1000
vy = vy*1000
theta = 240
vahw = 3.5 # half width of the total viewing angle width of the explorer probe in 2D
vahwr = vahw*np.pi/180 # same width expressed in radians
vsc = 30000 # velocity of spacecraft in m/s
vxshifted = np.array([]) # initializing arrays to store values
vyshifted = np.array([])
vxunshifted = np.array([])
vyunshifted = np.array([])
trackvangle = np.array([])
maxwcolorus = np.array([])
vsqshifted = np.array([])
thetarad = theta*np.pi/180 # expressing the value of theta in radians
# calculating the shift of the particle velocities into the spacecraft frame
xshiftfactor = -vsc*np.cos(thetarad + np.pi/2)
yshiftfactor = -vsc*np.sin(thetarad + np.pi/2)
vxshift = vx + xshiftfactor
vyshift = vy + yshiftfactor
vshifttotal = np.sqrt(vxshift**2 + vyshift**2)
vsquaredtotal = vxshift**2 + vyshift**2 # calculating total energies (v^2) associated with each trajectory in spacecraft frame
vangle = np.arccos(vxshift/vshifttotal) # calculating the new angle in which the velocity vector points for each trajectory
minangle1 = (thetarad - np.pi/2 - vahwr)
maxangle1 = (thetarad - np.pi/2 + vahwr)
if minangle1 < 0:
    minangle1 = minangle1 + 2*np.pi
if maxangle1 < 0:
    maxangle1 = maxangle1 + 2*np.pi
minangle2 = (thetarad + np.pi/2 - vahwr)
maxangle2 = (thetarad + np.pi/2 + vahwr)
if minangle2 > 2*np.pi:
    minangle2 = minangle2 - 2*np.pi
if maxangle2 > 2*np.pi:
    maxangle2 = maxangle2 - 2*np.pi
for i in range(vx.size):
    if vyshift[i] < 0:
        # accounting for angles below the x axis, which will have a cosine equal to the ones mirrored across the x axis
        vangle[i] = 2*np.pi - vangle[i]
    if minangle1 < vangle[i] and maxangle1 > vangle[i]:
        # appending values to the list of observable velocity shifted trajectories for trajectories in anti-ram
        vxshifted = np.append(vxshifted, vxshift[i])
        vyshifted = np.append(vyshifted, vyshift[i])
        vxunshifted = np.append(vxunshifted, vx[i])
        vyunshifted = np.append(vyunshifted, vy[i])
        trackvangle = np.append(trackvangle, vangle[i])
        maxwcolorus = np.append(maxwcolorus, f[i])
        vsqshifted = np.append(vsqshifted, vsquaredtotal[i])
    elif minangle2 < vangle[i] and maxangle2 > vangle[i]:
        # doing so for the set of trajectories in the ram direction
        vxshifted = np.append(vxshifted, vxshift[i])
        vyshifted = np.append(vyshifted, vyshift[i])
        vxunshifted = np.append(vxunshifted, vx[i])
        vyunshifted = np.append(vyunshifted, vy[i])
        trackvangle = np.append(trackvangle, vangle[i])
        maxwcolorus = np.append(maxwcolorus, f[i])
        vsqshifted = np.append(vsqshifted, vsquaredtotal[i])


# finding the slopes of the lines associated with the IBEX viewing angle
slope1 = np.tan(thetarad - np.pi/2 + vahwr)
slope2 = np.tan(thetarad - np.pi/2 - vahwr)

def viewangle1(x):
    #equation for the line of one boundary of the viewing angle
    return slope1*(x + xshiftfactor/1000) - yshiftfactor/1000

def viewangle2(x):
    # equation for the line of the other boundary of the viewing angle
    return slope2*(x + xshiftfactor/1000) - yshiftfactor/1000

# set of sample vx values to generate the lines
samplevxs = np.arange(-50, 50, 1)

esas = np.array([10, 19.44, 37.47, 72.83]) # value for ESA1's high energy boundary in keV

def eVtov(esaenergy):
    # converts energy in eV to velocity in m/s
    return np.sqrt(esaenergy*1.602*10**(-19)/(.5 * 1.6736*10**(-27)))

angleset1 = np.arange(thetarad - np.pi/2 - vahwr, thetarad - np.pi/2 + vahwr, .001)
angleset2 = np.arange(thetarad + np.pi/2 - vahwr, thetarad + np.pi/2 + vahwr, .001)



# plotting this set of trajectories
f2 = plt.figure()
f2.set_figwidth(10)
f2.set_figheight(6)
#plt.scatter(vxunshifted[:]/1000, vyunshifted[:]/1000, c=maxwcolorus[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-11)))
#plt.scatter(vx[:]/1000, vy[:]/1000, c=f[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-11)))
plt.scatter(vx[:]/1000, vy[:]/1000, c=f[:], marker='o', cmap='rainbow')
plt.plot(samplevxs, viewangle1(samplevxs), c='k')
plt.plot(samplevxs, viewangle2(samplevxs), c='k')
for i in range(esas.size):
    plt.plot(eVtov(esas[i])/1000*np.cos(angleset1) - xshiftfactor/1000, eVtov(esas[i])/1000*np.sin(angleset1) - yshiftfactor/1000, c='k')
    plt.plot(-eVtov(esas[i])/1000*np.cos(angleset1) - xshiftfactor/1000, -eVtov(esas[i])/1000*np.sin(angleset1) - yshiftfactor/1000, c='k')
plt.rcParams.update({'font.size': fsize})
cb = plt.colorbar()
cb.set_label('Normalized Phase Space Density')
plt.xlim([-50, 50])
plt.ylim([-45, 45])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
plt.show()


#######################################################################################################################################



# calculating the actual kinetic energy of each trajectory at the target point in eV
totalke = .5 * (1.6736*10**(-27)) * vsqshifted * 6.242*10**(18)

"""# plotting counts of energies for each observable trajectory
fig = plt.figure()
fig.set_figwidth(9)
fig.set_figheight(6)
# counts are weighted by value of the normalized phase space density
bincount = 100
binmeans, binedges, binnum = scipy.stats.binned_statistic(totalke, values=maxwcolorus, statistic=np.mean, bins=bincount)
binwidth = (binedges[1] - binedges[0])
bincenters = binedges[1:] - binwidth/2
plt.hist(totalke, bins=bincount, weights=maxwcolorus, log=True)
#plt.hlines(binmeans, binedges[:-1], binedges[1:], colors='r',lw=2, label='Mean PSD value in each bin')
#plt.hist(totalke, bins=100, weights=maxwcolorus)
#plt.yscale('log')
plt.axvline(x=10, c='k')
plt.axvline(x=19.44, c='k')
plt.axvline(x=37.47, c='k')
plt.ylim([10**(-12),10**(-5)])
plt.xlabel("Particle Energy at Target Point in eV")
plt.ylabel("Weighted Counts")
#plt.legend(fontsize=10)
plt.show()"""


#######################################################################################################################################

erangehigh = 10 # establishing boundaries for acceptable energies of particles in eV so we can probe specific energy regions
erangelow = 0
keselection = np.array([])
maxwcolorselect = np.array([])
vangleselect = np.array([])
for i in range(totalke.size):
    if erangelow < totalke[i] < erangehigh:
        # preserving trajectories in the appropriate energy region
        keselection = np.append(keselection, totalke[i])
        maxwcolorselect = np.append(maxwcolorselect, maxwcolorus[i])
        vangleselect = np.append(vangleselect, trackvangle[i])
# plotting trajectories in said energy range as a set of points on the unit circle according to where
# the spacecraft sees they come from
plt.scatter(-np.cos(vangleselect), -np.sin(vangleselect), c=maxwcolorselect, marker='o', cmap='rainbow', s=3, alpha=.5)
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.show()


#######################################################################################################################################


maxind = np.argmax(f)
print("Maximum value of the PSD is at velocity point (" + str(vx[maxind]) + ", " + str(vy[maxind]) + "), with PSD value " + str(f[maxind]))