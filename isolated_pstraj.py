import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from tqdm import tqdm

# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
oneyear = 3.15545454545*10**7

finalt = -1.5*oneyear # time to start backtracing
initialt = -1*10**(10) # time to backtrace to - should be at least -1*10**(10) to get all single cycle pseudo bounds
tstep = 10000 # general time resolution
tstepclose = 500 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 100 # upwind reference distance for backtraced trajectories, in au

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
# https://ibex.princeton.edu/sites/g/files/toruqf1596/files/moebius_et_al_2012.pdf
# above gives angle of ecliptic relative to ISM flow
sunpos = np.array([0,0,0]) # placing the Sun at the origin
theta = 120 # angle from upwind of target point location
ibexrad = 1 # radial distance of target point from the Sun
ibexx = ibexrad*np.cos(theta*np.pi/180)
ibexy = ibexrad*np.sin(theta*np.pi/180)
ibexpos = np.array([ibexx*au, ibexy*au, 0])
# implementation of target point that orbits around the Sun
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])


# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
# Initial Conditions for orbit starting at point of interest for backtracing
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]

# conditions for velocity at the target point to scan through
vxstart = np.arange(-60000, 40000, 300)
vystart = np.arange(-35000, 50000, 300)
vzstart = 0

tmid = finalt - 200000000 # time at which we switch from high resolution to low resolution - a little more than half of a cycle
# two resolutions to help ensure ionization integral doesn't blow up
tclose = np.arange(finalt, tmid, -tstepclose) # high resolution time array (close regime)
tfar = np.arange(tmid, initialt, -tstepfar) # low resolution time array (far regime)
t = np.concatenate((tclose, tfar))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# RADIATION PRESSURE FORCE FUNCTIONS
def LyaRP(t,v_r):
    # a double (triple) Gaussian function to mimic the Lyman-alpha profile
    lyafunction = 1.25*np.exp(-(v_r/1000-55)**2/(2*25**2)) + 1.25*np.exp(-(v_r/1000+55)**2/(2*25**2)) + .55*np.exp(-(v_r/1000)**2/(2*25**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match dividor in first term of addfactor
    scalefactor = 1.8956
    # added value to ensure scaling is throughout solar cycle
    # matches total irradiance out to +-120 km/s
    #addfactor = ((1.3244/1.616) - 1)*(.75 + .243*np.e)*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    # matches total irradiance out to +-370 km/s
    addfactor = ((1.55363/1.8956) - 1)*(.75 + .243*np.e)*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    return scalefactor*(.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + addfactor)*lyafunction

def LyaRP2(t,v_r):
    # Ly-a line profile function from Tarnopolski & Bzowski 2007
    lyafunction = np.e**(-3.8312*10**-5*(v_r/1000)**2)*(1 + .73879* \
    np.e**(.040396*(v_r/1000) - 3.5135*10**-4*(v_r/1000)**2) + .47817* \
    np.e**(-.046841*(v_r/1000) - 3.3373*10**-4*(v_r/1000)**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    # time dependent portion of the radiation pressure force function
    tdependence = 5.6*10**11 - np.e/(np.e + 1/np.e)*2.4*10**11 + 2.4*10**11/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))
    return 2.4543*10**-9*(1 + 4.5694*10**-4*tdependence)*lyafunction

# constants for following function
A_K = 6.523*(1 + 0.619)
m_K = 5.143*(1 -0.081)
del_K = 38.008*(1+0.104)
K = 2.165*(1-0.301)
A_R = 580.37*(1+0.28)
dm = -0.344*(1-0.828)
del_R = 32.349*(1-0.049)
b_bkg = 0.026*(1+0.184)
a_bkg = 0.411**(-4) *(1-1.333*0.0007)
#print(a_bkg)
r_E = 0.6
r2 = 1
def LyaRP3(t,v_r):
    #Author: E. Samoylov, H. Mueller LISM Group (Adapted by L. Dyke for this code)
    #Date: 04.18.2023
    #Purpose: To confirm the graph that EQ14 produces in
    #         Kowalska-Leszczynska's 2018 paper
    #         Evolution of the Solar Lyα Line Profile during the Solar Cycle
    #https://iopscience.iop.org/article/10.3847/1538-4357/aa9f2a/pdf
    F_R = A_R / (del_R * np.sqrt(2 * np.pi)) *np.exp(-(np.square((v_r/1000) - (m_K - dm))) / (2*(del_R ** 2)))
    F_bkg = np.add(a_bkg*(v_r/1000)*0.000001,b_bkg)
    F_K = A_K * np.power(1 + np.square((v_r/1000) - m_K) / (2 * K * ((del_K) ** 2)), -K - 1)

    omegat = 2*np.pi/(3.47*10**(8))*t
    # added value to ensure scaling is correct throughout solar cycle
    # matches total irradiance out to +-120 km/s
    #addfactor = ((.973/.9089) - 1)*.85*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    # matches total irradiance out to +-370 km/s
    addfactor = ((.97423/.91) - 1)*.85*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    # time dependent portion of the radiation pressure force function
    tdependence = .85 - np.e/(np.e + 1/np.e)*.33 + .33/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + addfactor
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match divisor in first term of addfactor
    scalefactor = .91
    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    return scalefactor*tdependence*(F_K-F_R+F_bkg)/(r_E/(r2**2))


def Lya_dr_dt(x,t,rp):
    # integrating differential equation for gravitational force. x[0:2] = x,y,z and x[3:5] = vx,vy,vz
    # dx0-2 = vx, vy, and vz, dx3-5 = ax, ay, and az
    r = np.sqrt((sunpos[0]-x[0])**2 + (sunpos[1]-x[1])**2 + (sunpos[2]-x[2])**2)
    # calculating the component of the radial unit vector in each direction at each point in time
    nrvecx = x[0]/r
    nrvecy = x[1]/r
    nrvecz = x[2]/r
    # calculating the magnitude of v_r at each point in time
    v_r = x[3]*nrvecx + x[4]*nrvecy + x[5]*nrvecz
    dx0 = x[3]
    dx1 = x[4]
    dx2 = x[5]
    # radiation pressure takes time and radial velocity as inputs
    radp = rp(t,v_r)
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-radp)
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-radp)
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-radp)
    return [dx0, dx1, dx2, dx3, dx4, dx5]


# flat Lyman-alpha profile function (only time dependence) and its associated function to input into odeint
def cosexprp(t):
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    return .75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))

def dr_dt(x,t,rp):
    # integrating differential equation for gravitational force. x[0:2] = x,y,z and x[3:5] = vx,vy,vz
    # dx0-2 = vx, vy, and vz, dx3-5 = ax, ay, and az
    r = np.sqrt((sunpos[0]-x[0])**2 + (sunpos[1]-x[1])**2 + (sunpos[2]-x[2])**2)
    dx0 = x[3]
    dx1 = x[4]
    dx2 = x[5]
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-rp(t))
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-rp(t))
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-rp(t))
    return [dx0, dx1, dx2, dx3, dx4, dx5]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# MAIN CODE
# code for tracking phase space at distance of x = 100 au away
# initializing arrays to store relevant values
farvx = np.array([])
farvy = np.array([])
fart = np.array([])
maxwcolor = np.array([])
vrmax = np.array([])
vrmin = np.array([])
backtraj = np.zeros((t.size, 6))
for i in tqdm(range(vxstart.size)): # displays progress bars for both loops to measure progress
    for j in tqdm(range(vystart.size)):
        # array of initial (final) phase space conditions at target point
        init = [xstart, ystart, zstart, vxstart[i], vystart[j], vzstart]
        # calculating the trajectory corresponding to said conditions
        # need to change between dr_dt and Lya_dr_dt depending on whether the profile is flat or not
        backtraj[:,:] = odeint(Lya_dr_dt, init, t, args=(LyaRP3,))
        if any(np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2) <= .00465*au):
            # tells the code to not consider the trajectory if it at any point intersects the width of the sun
            continue
        if all(backtraj[:,0]-sunpos[0] < refdist*au):
            # forgoes the following checks if the trajectory never passes through the plane at the reference distance upwind
            continue
        for k in range(t.size - tclose.size):
            if backtraj[k+tclose.size,0] >= refdist*au and backtraj[k-1+tclose.size,0] <= refdist*au:
                # adjusting the indexing to avoid checking in the close regime
                kn = k+tclose.size
                # only saving initial conditions corresponding to points that lie within this Maxwellian at reference distance
                # cutoff is 10^-3 of the core value BEFORE IONIZATION
                if np.sqrt((backtraj[kn-1,3]+26000)**2 + (backtraj[kn-1,4])**2 + (backtraj[kn-1,5])**2) <= 26795:
                    # approximate time-averaged charge exchange photoionization rate from Sokol et al. 2019
                    cxirate = 5*10**(-7)
                    # omega*t for each time point in the trajectory
                    omt = 2*np.pi/(3.47*10**(8))*t[0:kn+1]
                    # function for the photoionization rate at each point in time
                    PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
                    #PIrate2 = 1.21163*10**(-7) # time average of above
                    r1 = 1*au # reference radius
                    currentrad = np.sqrt((sunpos[0]-backtraj[0:kn+1,0])**2 + (sunpos[1]-backtraj[0:kn+1,1])**2 + (sunpos[2]-backtraj[0:kn+1,2])**2)
                    # calculating the component of the radial unit vector in each direction at each point in time
                    nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
                    nrvecy = (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
                    nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
                    # calculating the magnitude of v_r at each point in time
                    currentvr = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
                    # calculating the maximum and minimum v_r for the trajectory
                    vrmax = np.append(vrmax, max(currentvr))
                    vrmin = np.append(vrmin, min(currentvr))
                    # integrand for the photoionization and charge exchange ionization losses
                    btintegrand = PIrate2/currentvr*(r1/currentrad)**2 + cxirate/currentvr*(r1/currentrad)**2
                    # calculation of attenuation factor
                    attfact = scipy.integrate.simps(btintegrand, currentrad)
                    # retaining variables corresponding to vx, vy, t at the target point
                    farvx = np.append(farvx, [backtraj[0,3]])
                    farvy = np.append(farvy, [backtraj[0,4]])
                    fart = np.append(fart, [finalt - t[kn-1]])
                    # calculating value of phase space density based on the value at the crossing of x = 100 au
                    maxwcolor = np.append(maxwcolor, [np.exp(-np.abs(attfact))*np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2 + backtraj[kn-1,5]**2)/(10195)**2)])
                    break
                break

print('Finished')
#
#
#
#
#
#
#
#
#
# DATA WRITING/PLOTTING CODE
# writing data to a file - need to change each time or it will overwrite previous file
# will have to change this path manually to save elsewhere
# note that this will overwrite a file with the same name - be sure to change it between every unique run
file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/kowlyarp_2pi3_-1.5yrs_whole_cxi+cepi_tclose500_r=1au.txt", 'w')
#file = open("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p1fluccosexprp_35pi36_0y_direct_cosexppi_tclose400.txt", "w")
for i in range(farvx.size):
    # writes velocity components in km/s (v_x and v_y) and corresponding attenuated phase space density value
    file.write(str(farvx[i]/1000) + ',' + str(farvy[i]/1000) + ',' + str(maxwcolor[i]) + '\n')
file.close()

# plotting a scatterplot of vx and vy at the target point, colored by the phase space density
f = plt.figure()
f.set_figwidth(10) # setting plot width (inches)
f.set_figheight(6) # setting plot height (inches)
fsize = 18 # font size for plot
# plotting phase space points colored by value of attenuated phase space density value
plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='rainbow')
plt.rcParams.update({'font.size': fsize})
cb = plt.colorbar()
cb.set_label('Normalized Phase Space Density')
#plt.xlim([-25, 25]) # can manually adjust viewing limits of plot
#plt.ylim([-25, 25])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
# automated plot titles if need be
#plt.suptitle('VDF at target, at t $\\approx$ ' + str(round(finalt/(oneyear), 3)) + ' years, drawn from Maxwellian at 100 au centered on $v_x$ = -26 km/s')
#plt.title('Target at (' + str(round(ibexpos[0]/au, 3)) + ' au, ' + str(round(ibexpos[1]/au, 3)) + ' au), Time Resolution Close to Target = ' + str(tstepclose) + ' s')
plt.show()


# plotting a contour whose levels are values of the phase space density
"""f = plt.figure()
f.set_figwidth(9)
f.set_figheight(6)
levels = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0]
plt.tricontour(farvx[:]/1000, farvy[:]/1000, maxwcolor[:], levels)
cb = plt.colorbar()
cb.set_label('f(r,v,t)')
plt.xlabel("vx at Target in km/s")
plt.ylabel("vy at Target in km/s")
#plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
plt.suptitle('Phase space population at target (t = 6.246e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
#plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
plt.title('Target at (.707 au, .707 au)')
#plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
plt.show()"""


# section of code to calculate which trajectories could be observed by spacecraft - considers velocity shifts and viewing angle
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
vxshift = farvx - vsc*np.cos(thetarad - np.pi/2)
vyshift = farvy - vsc*np.sin(thetarad - np.pi/2)
vshifttotal = np.sqrt(vxshift**2 + vyshift**2)
vsquaredtotal = vxshift**2 + vyshift**2 # calculating total energies (v^2) associated with each trajectory in spacecraft frame
vangle = np.arccos(vxshift/vshifttotal) # calculating the new angle in which the velocity vector points for each trajectory
for i in range(farvx.size):
    if vyshift[i] < 0:
        # accounting for angles below the x axis, which will have a cosine equal to the ones mirrored across the x axis
        vangle[i] = 2*np.pi - vangle[i]
    if (thetarad + np.pi/2 - vahwr) < vangle[i] and (thetarad + np.pi/2 + vahwr) > vangle[i]:
        # appending values to the list of observable velocity shifted trajectories
        vxshifted = np.append(vxshifted, vxshift[i])
        vyshifted = np.append(vyshifted, vyshift[i])
        vxunshifted = np.append(vxunshifted, farvx[i])
        vyunshifted = np.append(vyunshifted, farvy[i])
        trackvangle = np.append(trackvangle, vangle[i])
        maxwcolorus = np.append(maxwcolorus, maxwcolor[i])
        vsqshifted = np.append(vsqshifted, vsquaredtotal[i])


# plotting this set of trajectories
f2 = plt.figure()
f2.set_figwidth(10)
f2.set_figheight(6)
plt.scatter(vxunshifted[:]/1000, vyunshifted[:]/1000, c=maxwcolorus[:], marker='o', cmap='rainbow')
plt.rcParams.update({'font.size': fsize})
cb = plt.colorbar()
cb.set_label('Normalized Phase Space Density')
#plt.xlim([-25, 25])
#plt.ylim([-25, 25])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
plt.show()

# calculating the actual kinetic energy of each trajectory at the target point in eV
totalke = .5 * (1.6736*10**(-27)) * vsqshifted * 6.242*10**(18)

# plotting counts of energies for each observable trajectory
fig = plt.figure()
fig.set_figwidth(8)
fig.set_figheight(5)
# counts are weighted by value of the normalized phase space density
plt.hist(totalke, bins=100, weights=maxwcolorus)
plt.xlabel("Particle Energy at Target Point in eV")
plt.ylabel("Weighted Counts")
plt.show()


erangehigh = 10 # establishing boundaries for acceptable energies of particles in eV so we can probe specific energy regions
erangelow = 1
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

overallvrmax = max(vrmax)
overallvrmin = min(vrmin)
meanvrmax = np.mean(vrmax)
meanvrmin = np.mean(vrmin)
# printing values of maximum/minimum radial velocities amongst all trajectories 
print("The maximum radial velocity out of all trajectories at all points in time is " + str(overallvrmax) + " m/s")
print("The minimum radial velocity out of all trajectories at all points in time is " + str(overallvrmin) + " m/s")
print("The mean of maximum radial velocities for all trajectories is: " + str(meanvrmax) + " m/s")
print("The mean of minimum radial velocities for all trajectories is: " + str(meanvrmin) + " m/s")
