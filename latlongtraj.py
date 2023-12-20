import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from tqdm import tqdm

#
# Setting common parameter values
# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
oneyear = 3.15545454545*10**7

# INITIAL CONDITIONS
finalt = 0 # time to start backtracing
initialt = -1*10**(10) # time to backtrace to
# two time resolutions to assist with ionization calculations
tstepclose = 500 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
refdist = 100 # upwind reference distance for backtraced trajectories, in au
# Here is where you can set the maximum latitude/longitude angles that you want to time the trajectory moving between
minlat = 80
maxlat = 100
minlong = 190
maxlong = 170
# converting to radians for the sake of the code
minlat = minlat*np.pi/180
maxlat = maxlat*np.pi/180
minlong = minlong*np.pi/180
maxlong = maxlong*np.pi/180

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
sunx = 0
suny = 0
sunz = 0
sunpos = np.array([sunx,suny,sunz])
# Angle of target point with respect to the upwind axis
theta = 120
# Radius from the sun of the target point
ibexrad = 1
# Calculating the x, y, z coordinates of the target point
# Note this currently approximates ISM Flow aligned with ecliptic
ibexx = ibexrad*np.cos(theta*np.pi/180)
ibexy = ibexrad*np.sin(theta*np.pi/180)
ibexpos = np.array([sunx + ibexx*au, suny + ibexy*au, sunz + 0])

tmid = finalt - 500000000 # time at which we switch from high resolution to low resolution - done to have high resolution close to Sun
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
#
# Set of radiation pressure force functions
def TGRP(t,v_r):
    # a triple Gaussian function to mimic the Lyman-alpha profile
    lyafunction = 1.25*np.exp(-(v_r/1000-55)**2/(2*25**2)) + 1.25*np.exp(-(v_r/1000+55)**2/(2*25**2)) + .55*np.exp(-(v_r/1000)**2/(2*25**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match dividor in first term of addfactor
    scalefactor = 1.8956
    # added value to ensure scaling is correct at both solar minimum and solar maximum
    # matches total irradiance out to +-120 km/s
    #addfactor = ((1.3244/1.616) - 1)*(.75 + .243*np.e)*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    # matches total irradiance out to +-370 km/s
    addfactor = ((1.55363/1.8956) - 1)*(.75 + .243*np.e)*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    return scalefactor*(.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + addfactor)*lyafunction

def TandBRP(t,v_r):
    # My Ly-a line profile function
    #lyafunction = 1.25*np.exp(-(v_r-55000)**2/(2*25000**2)) + 1.25*np.exp(-(v_r+55000)**2/(2*25000**2)) + .55*np.exp(-v_r**2/(2*25000**2))
    # Ly-a line profile function from Tarnopolski 2007
    lyafunction = np.e**(-3.8312*10**-5*(v_r/1000)**2)*(1 + .73879* \
    np.e**(.040396*(v_r/1000) - 3.5135*10**-4*(v_r/1000)**2) + .47817* \
    np.e**(-.046841*(v_r/1000) - 3.3373*10**-4*(v_r/1000)**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    # time dependent portion of the radiation pressure force function
    tdependence = 5.6*10**11 - np.e/(np.e + 1/np.e)*2.4*10**11 + 2.4*10**11/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))
    #return (.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))*lyafunction
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
def IKL2018RP(t,v_r):
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

    radp = rp(t,v_r)
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-radp)
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-radp)
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-radp)
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
# Main code
# Set of "initial" conditions (conditions at target point) of 
indxic = 10000
indyic = 2000
indzic = 00
init = [ibexpos[0], ibexpos[1], ibexpos[2], indxic, indyic, indzic]
print("Calculating trajectory...")
singletraj = odeint(Lya_dr_dt, init, t, args=(TandBRP,))
print("Trajectory Calculated")
trackrp = np.zeros(t.size) # array for tracking values of radiation pressure throughout trajectory
rtrack = np.sqrt((sunpos[0]-singletraj[:,0])**2 + (sunpos[1]-singletraj[:,1])**2 + (sunpos[2]-singletraj[:,2])**2)
# calculating the component of the radial unit vector in each direction at each point in time
nrvecxk = singletraj[:,0]/rtrack
nrvecyk = singletraj[:,1]/rtrack
nrveczk = singletraj[:,2]/rtrack
# calculating the magnitude of v_r at each point in time
v_rtrack = singletraj[:,3]*nrvecxk[:] + singletraj[:,4]*nrvecyk[:] + singletraj[:,5]*nrveczk[:]
# calculating the heliolatitude/heliolongitude angles at each point in the trajectory
# boolean mask to determine correct angle since cosine only covers half of x/y plane values
belowxaxis = singletraj[:,1] < 0
ymask = belowxaxis*2*np.pi
longmask = -2*(belowxaxis-.5) # -1 if below x axis in xy plane, 1 if above
# if y < 0, longitude = 2pi-arccos(x/r), otherwise longitude = arccos(x/r)
heliolong = ymask + np.arccos((singletraj[:,0] - sunx)/rtrack[:])*longmask
#print(min(heliolong))
#print(max(heliolong))
# same case with latitude
belowxyplane = singletraj[:,2] < 0
zmask = 2*(belowxyplane-.5) # 1 if below z axis, -1 if above
# if z < 0, latitude = pi/2 + arcsin(abs(z)/r), else latitude = pi/2 - arcsin(abs(z)/r)
heliolat = np.pi/2 + zmask*np.arcsin(np.abs(singletraj[:,2] - sunpos[2])/rtrack[:])
for k in tqdm(range(t.size)): # tqdm gives you a progress bar on this part of the code
    # NOTE YOU WILL HAVE TO CHANGE THE NEXT LINE MANUALLY WITH YOUR CHOICE OF RADIATION PRESSURE FORCE
    trackrp[k] = TandBRP(t[k],v_rtrack[k]) # calculating the value of the radiation pressure at each time point
    if np.sqrt((singletraj[k,0]-sunpos[0])**2 + (singletraj[k,1]-sunpos[1])**2 + (singletraj[k,2]-sunpos[2])**2) <= .00465*au:
        # checking if the orbit is too close to the sun
        print("Orbit too close to sun")
        # Setting variables if trajectory intersects with Sun so code doesn't throw an error
        psd = 0
        perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
        ttime = 0
        break
    if singletraj[k,0] >= refdist*au:
        # prints phase space data at time of crossing, and time of crossing
        print(singletraj[k-1,:])
        print(t[k-1])
        rprange = trackrp[:k]
        perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
        
        # approximate time-averaged charge exchange photoionization rate from Sokol et al. 2019
        cxirate = 5*10**(-7)
        omt = 2*np.pi/(3.47*10**(8))*t[0:k+1]
        # function for the photoionization rate at each point in time
        PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
        r1 = 1*au # reference radius
        currentrad = np.sqrt((sunpos[0]-singletraj[0:k+1,0])**2 + (sunpos[1]-singletraj[0:k+1,1])**2 + (sunpos[2]-singletraj[0:k+1,2])**2)
        # calculating the component of the radial unit vector in each direction at each point in time
        nrvecx = (-sunpos[0]+singletraj[0:k+1,0])/currentrad
        nrvecy= (-sunpos[1]+singletraj[0:k+1,1])/currentrad
        nrvecz = (-sunpos[2]+singletraj[0:k+1,2])/currentrad
        # calculating the magnitude of v_r at each point in time
        currentvr = singletraj[0:k+1,3]*nrvecx[0:k+1] + singletraj[0:k+1,4]*nrvecy[0:k+1] + singletraj[0:k+1,5]*nrvecz[0:k+1]
        # integrand for the photoionization losses
        btintegrand = PIrate2/currentvr*(r1/currentrad)**2 + cxirate/currentvr*(r1/currentrad)**2
        # calculation of attenuation factor
        attfact = scipy.integrate.simps(btintegrand, currentrad)
        psd = np.exp(-np.abs(attfact))*np.exp(-((singletraj[k-1,3]+26000)**2 + singletraj[k-1,4]**2 + singletraj[k-1,5]**2)/(5327)**2)

        #print(np.sqrt((singletraj[k-1,3]+26000)**2 + (singletraj[k-1,4])**2 + (singletraj[k-1,5])**2))
        print("Perihelion distance in au is: " + str(perihelion/au))
        print("PSD value: " + str(psd))
        
        t = t[:k]
        ttime = t[0] - t[-1]
        print("Travel time from 100 au in years: " + str(ttime/oneyear))
        heliolat = heliolat[:k] # clipping arrays to relevant range
        heliolong = heliolong[:k]
        inlongrange = (heliolong < maxlong) & (heliolong > minlong) # boolean array that equals True when in the right longitude range
        if maxlong < minlong:
            # considering case where the x-axis is in the longitudinal range
            inlongrange = ((heliolong > minlong) & (heliolong < 2*np.pi)) | ((heliolong < maxlong) & (heliolong > 0)) 
        timeinlong = np.array([]) # array to store lengths of time when the particle is in that range
        # applying the boolean mask to the time points to get the times where the trajectory is in the relevant range
        nonzerot = t*inlongrange
        ini = 0 # saving the index corresponding to the first time point to consider
        for i in tqdm(range(nonzerot.size-2)):
            # calculating time step between each array member
            cond1 = abs(nonzerot[i+2] - nonzerot[i+1])
            cond2 = abs(nonzerot[i+1] - nonzerot[i])
            if cond1 != cond2:
                # checks if there is an abnormal time step to split up different times of trajectory passing through region of interest
                if abs(cond1-cond2) != abs(tstepclose-tstepfar):
                    # double checks the difference isn't just a result of the regimes switching
                    timeinlong = np.append(timeinlong, abs(nonzerot[i+1]-nonzerot[ini]))
                    ini = i+2
            if i == nonzerot.size-3:
                #ensures end of array is considered
                timeinlong = np.append(timeinlong, abs(nonzerot[i+2]-nonzerot[ini]))
        timeinlong = timeinlong[np.where(timeinlong!=0)]
        #repeat the process for the heliolatitude
        inlatrange = (heliolat < maxlat) & (heliolat > minlat)
        nonzerotlat = t*inlatrange
        timeinlat = np.array([])
        ini2 = 0
        for i in tqdm(range(nonzerotlat.size-2)):
            cond1 = abs(nonzerotlat[i+2] - nonzerotlat[i+1])
            cond2 = abs(nonzerotlat[i+1] - nonzerotlat[i])
            if cond1 != cond2:
                if abs(cond1-cond2) != abs(tstepclose-tstepfar):
                    timeinlat = np.append(timeinlat, abs(nonzerotlat[i+1]-nonzerotlat[ini2]))
                    ini2 = i+2
            if i == nonzerotlat.size-3:
                timeinlat = np.append(timeinlat, abs(nonzerotlat[i+2]-nonzerotlat[ini2]))
        timeinlat = timeinlat[np.where(timeinlat!=0)]

        break
    if k == t.size-1:
        # Setting variables if the trajectory never reaches the injection plane so the code doesn't throw an error
        perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
        ttime = 0
        psd = 0

print("Total time(s) spent in the given range of longitudes: " + str(timeinlong/oneyear) + " years")
print("Total time(s) spent in the given range of latitudes: " + str(timeinlat/oneyear) + " years")
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
# Trajectory plotting code
zer = [0]
fosize = 10

# 2D plotting code
"""
# coloring the points along the trajectory by the value of the radiation pressure force
# limits the color bar between vmin and vmax
plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=0, vmax=2, zorder=2)
cb = plt.colorbar()
# plotting the location of the target point as a green dot
plt.scatter(ibexpos[0]/au, ibexpos[1]/au, c='springgreen', zorder=3)
# plotting the location of the Sun as an orange dot
plt.scatter(zer, zer, c='orange', zorder=3)
plt.grid(zorder=0)
cb.set_label("Value of $\mu$", fontsize=fosize)
cb.ax.tick_params(labelsize=fosize)
plt.xlabel("x (au)", fontsize=fosize)
plt.ylabel("y (au)", fontsize=fosize)
# determining limits of the plot
plt.xlim([-0.2,0.2])
plt.ylim([-0.2,0.2])

plt.xticks(fontsize=fosize)
plt.yticks(fontsize=fosize)
plt.title("Individual Orbit at time t$\\approx$" + str(round(finalt/(oneyear), 3)) + " years, Target at (" + str(round(ibexpos[0]/au, 4)) + " au, " + str(round(ibexpos[1]/au, 4)) + " au, " + str(round(ibexpos[2]/au, 4)) + " au) \
    \n At target point v = (" + str(indxic/1000) + " km/s, " + str(indyic/1000) + " km/s, " + str(indzic/1000) + " km/s), Value of distribution function = " + str(psd) + " \
    \n Perihelion at $\\sim$ " + str(round(perihelion/au, 5)) + " au, Travel time from x = 100 au to target $\\approx$ " + str(round(ttime/oneyear, 3)) + " years", fontsize=10)
plt.show()"""

# 3D plotting code
"""fig3d = plt.figure()
fig3d.set_figwidth(7)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
# plotting the trajectory as a series of scatter plot points colored by radiation pressure
# limits the color bar between vmin and vmax
scatterplot = ax3d.scatter3D(singletraj[:,0]/au, singletraj[:,1]/au, singletraj[:,2]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=(.75-.243/np.e), vmax=(.75+.243*np.e))
cb = fig3d.colorbar(scatterplot)
cb.set_label('Value of mu')
# plotting the location of the Sun as an orange dot
ax3d.scatter3D(zer,zer,zer,c='orange')
# plotting the location of the target point as a green dot
ax3d.scatter3D([ibexpos[0]/au],[ibexpos[1]/au],[ibexpos[2]/au], c='springgreen')
ax3d.set_xlabel("x (au)")
ax3d.set_ylabel("y (au)")
ax3d.set_zlabel("z (au)")
# determining limits of the view
ax3d.set_xlim3d(left = -1.5, right = 1)
ax3d.set_ylim3d(bottom = -1, top = 1)
ax3d.set_zlim3d(bottom = -1, top = 1)
# setting the initial viewing angle
ax3d.view_init(90,270)
ax3d.set_title("Individual Orbit at time t$\\approx$" + str(round(finalt/(oneyear), 3)) + " years \n Target at (" + str(round(ibexpos[0]/au, 4)) + " au, " + str(round(ibexpos[1]/au, 4)) + " au, " + str(round(ibexpos[2]/au, 4)) + " au) \
    \n At target point v = (" + str(indxic/1000) + " km/s, " + str(indyic/1000) + " km/s, " + str(indzic/1000) + " km/s) \
    \n Value of distribution function = " + str(psd) + "\
    \n Perihelion at $\\sim$ " + str(round(perihelion/au, 5)) + " au \
    \n Travel time from x = 100 au to target $\\approx$ " + str(round(ttime/oneyear, 3)) + " years",fontsize=12)"""