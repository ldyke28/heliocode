import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
import scipy.interpolate
from tqdm import tqdm
import h5py
from scipy.signal import butter, lfilter, freqz

#####################################################################################################################################
# CODE INTRODUCTION
#####################################################################################################################################

"""
This is the serial version of the code that runs in 2D, which has been updated and much of the old, no longer used fluff removed. 
It is a trajectory calculating code that considers the motion of ISN H according to the equations of motion in the heliosphere, which 
involves gravity and a radiation pressure force that is dependent on the time in the solar cycle, the position of the particle 
relative to the Sun, and its velocity. Here are the processes the code completes:

1. Define relevant fixed values used in the calculation, and define initial conditions for the specific run
2. Filters and interpolates the LASP solar irradiance data throughout multiple solar cycles to use in the calculation of the 
radiation pressure force
3. Imports the distributions on the boundary surface, which will determine the value of the PSD for particles reaching the boundary

FOR MODE 2 (individual orbit plotting code):
4. Calculates the individual trajectory, calculating the ionization losses from charge exchange and photoionization for it, using the
boundary distributions to inform the initial value of the PSD
5. Plots the trajectory in 2D in the simulation plane within a range in phase space that is controlled manually

FOR MODE 3 (phase space plotting):
4. Calculates each trajectory, calculating the ionization losses from charge exchange and photoionization for them, using the
boundary distributions to inform each initial value of the PSD
5. Saves the trajectory data to a file in the form of (vx, vy, PSD)
6. Plots the points as a scatter plot in 2D phase space colored by the value of the attenuated PSD
"""

#####################################################################################################################################






# MODE LIST
# 2 = plot an individual trajectory traced backward from point of interest
# 3 = generate phase space diagram
mode = 3




# RELEVANT (FIXED) VARIABLES THROUGHOUT THE CODE
#####################################################################################################################################
# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
mH = 1.6736*10**(-27) # mass of hydrogen in kg
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
oneyear = 3.15545454545*10**7

# 120749800 for first force free
# 226250200 for second force free
finalt = 0*oneyear # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -5*10**(10) # time in the past to which the code should backtrace
tstep = 10000 # general time resolution
tstepclose = 300 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 70 # reference distance for backtraced trajectories, in au

theta = 275 # angle with respect to upwind axis of target point in degrees
ibexrad = 1 # radial distance of target point from Sun

vsolarwindms = 400000 # solar wind speed in m/s
vsolarwindcms = vsolarwindms*100 # solar wind speed in cm/s
nsw1au = 5 # solar wind density at 1 au in cm^-3

r1 = 1*au # reference radius for ionization integration

noncalcextent = 15 # radius in km/s around the axis of exclusion where we don't want to calculate trajectories (to save computational time)

# path to the file to which output is written
fpath = "C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/1stoffsetfilterfulltdrp_-17pi36_0yr_whole_newcx+pi_tclose300_r=1au_interpdist.txt"

#####################################################################################################################################
# ESTABLISHING INITIAL CONDITIONS FOR THE CODE
#####################################################################################################################################

# Multiple sets of initial vx/vy conditions for convenience
# vx/vy initial conditions are sampled on a grid with chosen resolution
#vxstart = np.arange(-36300, -29200, 20)
#vystart = np.arange(-6600, -4600, 20)
#vxstart = np.arange(-25000, 25000, 300)
#vystart = np.arange(-25000, 25000, 300)
#vxstart = np.arange(-60000, 40000, 300)
#vystart = np.arange(-35000, 50000, 300)
vxstart = np.arange(-50000, 50000, 500)
vystart = np.arange(-45000, 45000, 500)
vzstart = 0

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
ibexx = ibexrad*np.cos(theta*np.pi/180)
ibexy = ibexrad*np.sin(theta*np.pi/180)
ibexpos = np.array([ibexx*au, ibexy*au, 0])
# implementation of target point that orbits around the sun
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])


# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
ttotal = 7000000000
if mode==2:
    startt = finalt
    lastt = initialt
    tmid = startt - 500000000 # time at which we switch from high resolution to low resolution - a little more than half of a solar cycle
    tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
    tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
    t = np.concatenate((tclose, tfar))
tscale = int(.7*ttotal/tstep)
#tscale = 0


# Initial Conditions for orbit starting at point of interest for backtracing
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]

if mode==3:
    startt = finalt
    lastt = initialt
    tmid = startt - 5.5*oneyear # time at which we switch from high resolution to low resolution - a little more than half of a solar cycle
    #countoffset = int((11*oneyear) / (tstepclose) / 7)
    countoffset = 0
    tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
    tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
    t = np.concatenate((tclose, tfar))
    mode3dt = startt-lastt
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
#

# FILTERING AND INTERPOLATING THE LASP IRRADIANCE DATA TO USE FOR THE RADIATION PRESSURE FORCE
#####################################################################################################################################

irradfile = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Time Dependent Irradiance Data/complya.csv", delimiter=',')

day = irradfile[:,0]
irradiance = irradfile[:,1]

seconds = day*86400

N = 28

oldirradianceavg = np.zeros(seconds.size)
for i in tqdm(range(seconds.size)):
    avgnumerator = 0
    lowerbound = -(int(N/2))
    upperbound = int(N/2)
    if i + lowerbound < 0:
        lowerbound = -i
    if i + upperbound >= seconds.size:
        upperbound = seconds.size - i - 1
    for j in range(-lowerbound):
        avgnumerator += irradiance[i+lowerbound]
    avgnumerator += irradiance[i]
    for k in range(upperbound):
        avgnumerator += irradiance[i+upperbound]
    oldirradianceavg[i] = avgnumerator/(upperbound - lowerbound + 1)

irradianceavg = np.array([])
secondsnew = np.array([])
for i in range(seconds.size):
    if (i+1) % N == 0:
        irradianceavg = np.append(irradianceavg, [oldirradianceavg[i]])
        secondsnew = np.append(secondsnew, [seconds[i]])

secondsnew = secondsnew - 1.9*10**9
seconds = seconds - 1.9*10**9
secondstoyears = 1/(86400*365)

wm2toph = 6.12*10**(13)

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 5
#fs = .000002       # sample rate, Hz
fs = 1/(86400)
cutoff = 1/(1.577*10**8)  # desired cutoff frequency of the filter, Hz
offset = np.mean(irradiance[0:10])

irradianceoffset = irradiance - offset

filterediaoffset = butter_lowpass_filter(irradianceoffset, cutoff, fs, order)

filteredia = filterediaoffset + offset

tgrid = np.meshgrid(secondsnew, indexing='ij') # order will be z, y, x for this

fifthorderoffset = 2.5*oneyear
firstorderoffset = .5*oneyear

irradianceinterp = scipy.interpolate.RegularGridInterpolator(points=[seconds-fifthorderoffset], values=filteredia)

#####################################################################################################################################

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
#

# IMPORTING AND INTERPOLATING BOUNDARY DISTRIBUTIONS FROM THE UAH GROUP'S FRAMEWORK
#####################################################################################################################################

datafilename1 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_001_H_RegAll.h5'
datafilename2 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_002_H_RegAll.h5'
datafilename3 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_003_H_RegAll.h5'
datafilename4 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_004_H_RegAll.h5'
datafilename5 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_005_H_RegAll.h5'
datafilename6 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_006_H_RegAll.h5'
datafilename7 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_007_H_RegAll.h5'
datafilename8 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_008_H_RegAll.h5'

with h5py.File(datafilename1, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array

    print(list(f.keys()))
    dsvdf1 = f['VDF3D'] # returns h5py dataset object
    arrvdf1 = f['VDF3D'][()] # returns np.array of values
    #print(arrvdf[128,128,128])
    xloc = f['vx_grid'][()]
    yloc = f['vy_grid'][()]
    zloc = f['vz_grid'][()]

with h5py.File(datafilename2, "r") as f:
    dsvdf2 = f['VDF3D'] # returns h5py dataset object
    arrvdf2 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename3, "r") as f:
    dsvdf3 = f['VDF3D'] # returns h5py dataset object
    arrvdf3 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename4, "r") as f:
    dsvdf4 = f['VDF3D'] # returns h5py dataset object
    arrvdf4 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename5, "r") as f:
    dsvdf5 = f['VDF3D'] # returns h5py dataset object
    arrvdf5 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename6, "r") as f:
    dsvdf6 = f['VDF3D'] # returns h5py dataset object
    arrvdf6 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename7, "r") as f:
    dsvdf7 = f['VDF3D'] # returns h5py dataset object
    arrvdf7 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename8, "r") as f:
    dsvdf8 = f['VDF3D'] # returns h5py dataset object
    arrvdf8 = f['VDF3D'][()] # returns np.array of values

zgrid, ygrid, xgrid = np.meshgrid(zloc, yloc, xloc, indexing='ij') # order will be z, y, x for this

interp1 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf1)
interp2 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf2)
interp3 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf3)
interp4 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf4)
interp5 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf5)
interp6 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf6)
interp7 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf7)
interp8 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf8)

#####################################################################################################################################

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
#

# DEFINING FUNCTIONS TO HANDLE THE RADIATION PRESSURE FORCE AND TO INPUT TO THE odeint FUNCTION
#####################################################################################################################################

def lya_abs_update(t,x,y,z,vr):
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    r = np.sqrt(x**2 + y**2 + z**2)
    rxy = np.sqrt(x**2 + y**2)
    # calculating the latitudinal (polar) angle in 3D space
    # since sine/cosine only covers half of the space, we have to manually check where the point is to get the right angle
    if z >= 0:
        latangle = np.pi/2 - np.arcsin(z/r)
    else:
        latangle = np.pi/2 + np.arcsin(np.abs(z)/r)
    # calculating the longitudinal (azimuthal) angle in 3D space
    if y >= 0:
        longangle = np.arccos(x/rxy)
    else:
        longangle = 2*np.pi - np.arccos(x/rxy)
    longangle = longangle - np.pi
    if longangle < 0:
        longangle = 2*np.pi + longangle
    latangled = latangle*180/np.pi
    longangled = longangle*180/np.pi

    alpha = .07 # alpha for the skew gaussian distribution
    
    # calculating parameters from IKL et al. 2022 paper: https://ui.adsabs.harvard.edu/abs/2022ApJ...926...27K/abstract
    # manually fitted based on Figure 3
    # amplitude
    if r < au:
        amp = 0
    else:
        amp = ((.59*(r/au - 12)/np.sqrt((r/au - 12)**2 + 200) + 0.38) + -0.4* \
        np.e**(-(180 - longangled)**2/50**2 - (r/au - 31)**2/15**2)*(1 + \
        scipy.special.erf(alpha*(r/au)/np.sqrt(2)))*(1 - np.e**(-(r/au)/4)))*1/.966

    # mean Doppler shift
    mds = -20*np.cos(longangle)*np.cos((latangled-100)*np.pi/180)
    # dispersion (width of the peak)
    disper = -.0006947*(r/au)**2 + .1745*(r/au) + 5.402 + \
        1.2*np.e**(-(longangled - 5)**2/50**2 - ((r/au) - 80)**2/60**2) + \
        3*np.e**(-(longangled - 180)**2/50**2 - ((r/au))**2/5**2) + \
        1*np.e**(-(longangled - 170)**2/50**2 - ((r/au) - 25)**2/200**2) + \
        .35*np.cos(((latangled - 75)*np.pi/180)*2)
    # fit exponent
    if r >= 50*au:
        fittype = 4
    else:
        fittype = 2
    # calculating equation 12 from the 2022 paper
    absval = amp*np.exp(-.5 * ((vr/1000 - mds)/disper)**fittype)

    # time dependent portion of the radiation pressure force function
    #tdependence = .85 - np.e/(np.e + 1/np.e)*.33 + .33/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + addfactor
    tdependence = .95 + .5/(np.e**2 + 1) + .5/(np.e + 1/np.e)*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))

    # flattening factor for heliolatitude dependence of radiation pressure force
    alya = .8
    Itotavg = 0.006944444
    latdep = np.sqrt(alya*np.sin(latangle)**2 + np.cos(latangle)**2)
    ttemp = t
    tbounded = False
    while not tbounded:
        if ttemp >= -1.392*10**(9):
            tbounded = True
        else:
            ttemp = ttemp + 1.392*10**(9)
    tdependence = irradianceinterp([ttemp])*latdep/Itotavg
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match divisor in first term of addfactor
    scalefactor = .333
    
    # parameters of function
    A_K = 6.523*(1 + 0.619*tdependence)
    m_K = 5.143*(1 - 1.081*tdependence)
    del_K = 38.008*(1 + 0.104*tdependence)
    K = 2.165*(1 - 0.301*tdependence)
    A_R = 580.37*(1 + 0.28*tdependence)
    dm = -0.344*(1 - 0.828*tdependence)
    del_R = 32.349*(1 - 0.049*tdependence)
    b_bkg = 0.035*(1 + 0.184*tdependence)
    a_bkg = 0.411**(-4) *(1 - 1.333*tdependence)
    #print(a_bkg)
    r_E = 0.6
    r2 = 1
    F_R = A_R / (del_R * np.sqrt(2 * np.pi)) *np.exp(-(np.square((vr/1000) - (m_K + dm))) / (2*(del_R ** 2)))
    F_bkg = np.add(a_bkg*(vr/1000)*0.000001,b_bkg)
    F_K = A_K * np.power(1 + np.square((vr/1000) - m_K) / (2 * K * ((del_K) ** 2)), -K - 1)

    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    return scalefactor*(F_K-F_R+F_bkg)/(r_E**2/(r2**2))*(1 - absval)


# odeint documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

def Abs_dr_dt(x,t,rp):
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
    radp = rp(t,x[0],x[1],x[2],v_r)
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-radp)
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-radp)
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-radp)
    return [dx0, dx1, dx2, dx3, dx4, dx5]

#####################################################################################################################################

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
# single trajectory plotting code
if mode==2:
    indxic = -22037
    indyic = 311
    indzic = 00
    init = [ibexpos[0], ibexpos[1], ibexpos[2], indxic, indyic, indzic]
    print("Calculating trajectory...")
    # calculating the trajectory given the initial conditions
    singletraj = odeint(Abs_dr_dt, init, t, args=(lya_abs_update,))
    print("Trajectory Calculated")
    # initializing arrays to store relevant values
    trackrp = np.zeros(t.size)
    Ltrack = np.zeros(t.size)
    Evartrack = np.zeros(t.size)
    Etrack = np.zeros(t.size)
    # getting the full array of radial distance values
    rtrack = np.sqrt((sunpos[0]-singletraj[:,0])**2 + (sunpos[1]-singletraj[:,1])**2 + (sunpos[2]-singletraj[:,2])**2)
    # calculating the component of the radial unit vector in each direction at each point in time
    nrvecxk = singletraj[:,0]/rtrack
    nrvecyk = singletraj[:,1]/rtrack
    nrveczk = singletraj[:,2]/rtrack
    # calculating the magnitude of v_r at each point in time
    v_rtrack = singletraj[:,3]*nrvecxk[:] + singletraj[:,4]*nrvecyk[:] + singletraj[:,5]*nrveczk[:]
    for k in tqdm(range(t.size)):
        trackrp[k] = lya_abs_update(t[k], singletraj[k,0], singletraj[k,1], singletraj[k,2], v_rtrack[k]) # calculating the value of the radiation pressure at each time point
        #trackrp[k] = rp2(t[k])
        # outdated code to track angular momentum/energy along the path
        """rmag = np.sqrt((sunpos[0]-singletraj[k,0])**2 + (sunpos[1]-singletraj[k,1])**2 + (sunpos[2]-singletraj[k,2])**2)
        rtrack[k] = rmag
        vmag = np.sqrt(singletraj[k,3]**2 + singletraj[k,4]**2 + singletraj[k,5]**2)

        rvec = (singletraj[k,0:3]-sunpos)/rmag
        rxv = np.cross(singletraj[k,0:3], singletraj[k,3:6])
        Ltrack[k] = np.sqrt(rxv[0]**2 + rxv[1]**2 + rxv[2]**2)

        vdotr = rvec[0]*singletraj[k,3] + rvec[1]*singletraj[k,4] + rvec[2]*singletraj[k,5]
        Evartrack[k] = Evartrack[k-1] + (t[k]-t[k-1])*rp6(t[k])*vdotr/(rmag**2)
        Etrack[k] = (vmag**2)/2 - G*msolar/rmag - G*msolar*Evartrack[k]"""
        # current value of the radial distance
        strajrad = np.sqrt((singletraj[k,0]-sunpos[0])**2 + (singletraj[k,1]-sunpos[1])**2 + (singletraj[k,2]-sunpos[2])**2)
        if strajrad <= .00465*au:
            # checking if the orbit is too close to the sun
            print("Orbit too close to sun")
            psd = 0
            perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
            ttime = 0
            break
        if strajrad >= refdist*au:
            #print(singletraj[k-1,:])
            #print(t[k-1])
            Etrack = Etrack[:k]
            rtrack = rtrack[:k]
            Ltrack = Ltrack[:k]
            rprange = trackrp[:k]
            # calculating the perihelion of the orbit
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

            # calculation of heliographic latitude angle (polar angle)
            belowxy = singletraj[0:k+1,2] < 0
            zmask = 2*(belowxy-.5)
            latangle = np.pi/2 + zmask*np.arcsin(np.abs(singletraj[0:k+1,2] - sunpos[2])/currentrad[:])
            
            # calculation of attenuation factor based on heliographic latitude angle
            btintegrand = btintegrand*(.85*(np.sin(latangle))**2 + (np.cos(latangle))**2)
            # calculation of attenuation factor
            attfact = scipy.integrate.simps(btintegrand, currentrad)
            # pre-attenuation value of normalized phase space density from pristine LISM Maxwellian 
            initpsd = np.exp(-((singletraj[k+1,3]+26000)**2 + singletraj[k+1,4]**2 + singletraj[k+1,5]**2)/(10195)**2)
            # attenuated NPSD value
            psd = np.exp(-np.abs(attfact))*initpsd


            updatedprofile = True
            if updatedprofile:
                endradxy = np.sqrt((sunpos[0]-singletraj[k,0])**2 + (sunpos[1]-singletraj[k,1])**2)
                belowxaxis = singletraj[k,1] < 0
                ymask = belowxaxis*2*np.pi
                longmask = -2*(belowxaxis-.5) # -1 if below x axis in xy plane, 1 if above
                # if y < 0, longitude = 2pi-arccos(x/r), otherwise longitude = arccos(x/r)
                endlongangle = ymask + np.arccos((singletraj[k,0] - sunpos[0])/endradxy)*longmask
                endlongangle = endlongangle*180/np.pi
                # finding the initial value of the distribution function based on the interpolated distributions
                endvelcoords = [singletraj[k,5]/1000,singletraj[k,4]/1000,singletraj[k,3]/1000]
                if endlongangle >= 0 and endlongangle <= 45:
                    anglepct = (endlongangle)/45
                    initpsd = interp1(endvelcoords)*(1 - anglepct) + interp2(endvelcoords)*anglepct
                elif endlongangle > 45 and endlongangle <= 90:
                    anglepct = (endlongangle - 45)/45
                    initpsd = interp2(endvelcoords)*(1 - anglepct) + interp3(endvelcoords)*anglepct
                elif endlongangle > 90 and endlongangle <= 135:
                    anglepct = (endlongangle - 90)/45
                    initpsd = interp3(endvelcoords)*(1 - anglepct) + interp4(endvelcoords)*anglepct
                elif endlongangle > 135 and endlongangle <= 180:
                    anglepct = (endlongangle - 135)/45
                    initpsd = interp4(endvelcoords)*(1 - anglepct) + interp5(endvelcoords)*anglepct
                elif endlongangle > 180 and endlongangle <= 225:
                    anglepct = (endlongangle - 180)/45
                    initpsd = interp5(endvelcoords)*(1 - anglepct) + interp6(endvelcoords)*anglepct
                elif endlongangle > 225 and endlongangle <= 270:
                    anglepct = (endlongangle - 225)/45
                    initpsd = interp6(endvelcoords)*(1 - anglepct) + interp7(endvelcoords)*anglepct
                elif endlongangle > 270 and endlongangle <= 315:
                    anglepct = (endlongangle - 270)/45
                    initpsd = interp7(endvelcoords)*(1 - anglepct) + interp8(endvelcoords)*anglepct
                elif endlongangle > 315 and endlongangle <= 360:
                    anglepct = (endlongangle - 315)/45
                    initpsd = interp8(endvelcoords)*(1 - anglepct) + interp1(endvelcoords)*anglepct
                
                print("Initial PSD value: " + str(initpsd[0]))

                vsolarwindms = 400000 # solar wind speed in m/s
                vsolarwindcms = vsolarwindms*100 # solar wind speed in cm/s
                nsw1au = 5 # solar wind density at 1 au in cm^-3

                r1 = 1*au # reference radius
                # radial distance to the Sun at all points throughout the trajectory
                currentrad = np.sqrt((sunpos[0]-singletraj[0:k,0])**2 + (sunpos[1]-singletraj[0:k,1])**2 + (sunpos[2]-singletraj[0:k,2])**2)

                # velocity squared at each point in time for the trajectory
                currentvsq = np.square(singletraj[0:k,3]) + np.square(singletraj[0:k,4]) + np.square(singletraj[0:k,5])
                # thermal velocity (temperature taken from Federico's given temperature)
                vth = np.sqrt(2 * 1.381*10**(-23) * 7500 / (1.672*10**(-27)))
                # omega for the relative velocity
                omegavs = np.abs(np.sqrt(currentvsq) - vsolarwindms)/vth
                # calculating the current collision velocity of the particle and the SW
                currentvrel = vth*(np.exp(-omegavs**2)/np.sqrt(np.pi) + (omegavs + 1/(2*omegavs))*scipy.special.erf(omegavs))
                # calculating kinetic energy in keV at each point in time
                currentKE = (.5 * mH * np.square(currentvrel)) * 6.242*10**(15)

                # parameters for function to calculate charge exchange cross section
                a1 = 4.049
                a2 = 0.447
                a3 = 60.5
                # function for H-H+ charge exchange cross section, from Swaczyna et al. (2019)
                # result is in cm^2
                # https://ui.adsabs.harvard.edu/abs/2019AGUFMSH51C3344S/abstract
                cxcrosssection = ((a1 - a2*np.log(currentKE))**2 * (1 - np.exp(-a3 / currentKE))**(4.5))*10**(-16)

                #nsw = nsw1au * (r1/currentrad)**2 # assuming r^2 falloff for density (Sokol et al. 2019)

                cxirate = nsw1au * currentvrel*100 * cxcrosssection

                #print(cxirate)

                # approximate time-averaged charge exchange photoionization rate from Sokol et al. 2019
                #cxirate = 5*10**(-7)
                # omega*t for each time point in the trajectory
                omt = 2*np.pi/(3.47*10**(8))*t[0:k]
                # function for the photoionization rate at each point in time
                PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
                #PIrate2 = 1.21163*10**(-7) # time average of above
                # calculating the component of the radial unit vector in each direction at each point in time
                nrvecx = (-sunpos[0]+singletraj[0:k,0])/currentrad
                nrvecy = (-sunpos[1]+singletraj[0:k,1])/currentrad
                nrvecz = (-sunpos[2]+singletraj[0:k,2])/currentrad
                # calculating the magnitude of v_r at each point in time
                currentvr = singletraj[0:k,3]*nrvecx[0:k] + singletraj[0:k,4]*nrvecy[0:k] + singletraj[0:k,5]*nrvecz[0:k]
                # calculating maximum and minimum radial velocities throughout the whole trajectory
                #vrmax = np.append(vrmax, max(currentvr))
                #vrmin = np.append(vrmin, min(currentvr))
                # integrand for the photoionization and charge exchange ionization losses
                btintegrand = PIrate2/currentvr*(r1/currentrad)**2 + cxirate/currentvr*(r1/currentrad)**2

                # calculation of heliographic latitude angle (polar angle)
                #belowxy = singletraj[0:k,2] < 0
                #zmask = 2*(belowxy-.5)
                #latangle = np.pi/2 + zmask*np.arcsin(np.abs(singletraj[0:k,2] - sunpos[2])/currentrad[:])
                
                # calculation of attenuation factor based on heliographic latitude angle
                #btintegrand = btintegrand*(.85*(np.sin(latangle))**2 + (np.cos(latangle))**2)
                # calculation of attenuation factor
                attfact = scipy.integrate.simps(btintegrand, currentrad)
                # calculating the value of the phase space density after attenuation
                psd = np.exp(-np.abs(attfact))*initpsd[0]
            print(np.sqrt((singletraj[k-1,3]+26000)**2 + (singletraj[k-1,4])**2 + (singletraj[k-1,5])**2))
            print("Velocity components in x and y: " + str(singletraj[k-1,3]) + ", " + str(singletraj[k-1,4]))
            print("Perihelion distance in au is: " + str(perihelion/au))
            print("PSD value: " + str(psd))
            
            t = t[:k] # clipping array of time points
            ttime = t[0] - t[-1] # calculating travel time from boundary surface to target point
            print("Travel time from " + str(refdist) + " au in years: " + str(ttime/oneyear))
            break
        if k == t.size-1:
            perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
            ttime = 0
            psd = 0

if mode==2:
    # lots of plotting code to plot the orbit on its own in 3D (above) or 2D (below)
    zer = [0]
    fosize = 10
    """fig3d = plt.figure()
    fig3d.set_figwidth(7)
    fig3d.set_figheight(7)
    ax3d = plt.axes(projection='3d')
    # plotting the trajectory as a series of scatter plot points colored by radiation pressure
    scatterplot = ax3d.scatter3D(singletraj[:,0]/au, singletraj[:,1]/au, singletraj[:,2]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=(.75-.243/np.e), vmax=(.75+.243*np.e))
    cb = fig3d.colorbar(scatterplot)
    cb.set_label('Value of mu')
    #ax3d.plot3D(trajs[:,0,1], trajs[:,1,1], trajs[:,2,1], 'gold', linestyle='--')
    #ax3d.plot3D(trajs[:,0,2], trajs[:,1,2], trajs[:,2,2], 'forestgreen', linestyle=':')
    #ax3d.plot3D(trajs[:,0,3], trajs[:,1,3], trajs[:,2,3], 'firebrick', linestyle='-.')
    ax3d.scatter3D(zer,zer,zer,c='orange')
    ax3d.scatter3D([ibexpos[0]/au],[ibexpos[1]/au],[ibexpos[2]/au], c='springgreen')
    ax3d.set_xlabel("x (au)")
    ax3d.set_ylabel("y (au)")
    ax3d.set_zlabel("z (au)")
    ax3d.set_xlim3d(left = -1.5, right = 1)
    ax3d.set_ylim3d(bottom = -1, top = 1)
    ax3d.set_zlim3d(bottom = -1, top = 1)
    ax3d.view_init(90,270)
    ax3d.set_title("Individual Orbit at time t$\\approx$" + str(round(finalt/(oneyear), 3)) + " years \n Target at (" + str(round(ibexpos[0]/au, 4)) + " au, " + str(round(ibexpos[1]/au, 4)) + " au, " + str(round(ibexpos[2]/au, 4)) + " au) \
        \n At target point v = (" + str(indxic/1000) + " km/s, " + str(indyic/1000) + " km/s, " + str(indzic/1000) + " km/s) \
        \n Value of distribution function = " + str(psd) + "\
        \n Perihelion at $\\sim$ " + str(round(perihelion/au, 5)) + " au \
        \n Travel time from x = 100 au to target $\\approx$ " + str(round(ttime/oneyear, 3)) + " years",fontsize=12)"""
    fig = plt.figure()
    fig.set_figwidth(9)
    fig.set_figheight(6)
    
    # for fluctuating force
    # plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=((.75-.243/np.e)-.1), vmax=((.75+.243*np.e)+.1), zorder=2)
    # for non-fluctuating force
    # vmin = minimum value of radiation pressure force, vmax = maximum value of radiation pressure force
    #plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=((.75-.243/np.e)), vmax=((.75+.243*np.e)), zorder=2)
    plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=0, vmax=2, zorder=2)
    #plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=min(rprange), zorder=2)
    cb = plt.colorbar()
    plt.scatter(ibexpos[0]/au, ibexpos[1]/au, c='springgreen', zorder=3) # plotting the target point
    plt.scatter(zer, zer, c='orange', zorder=3) # plotting the Sun
    plt.grid(zorder=0)
    cb.set_label("Value of $\mu$", fontsize=fosize)
    cb.ax.tick_params(labelsize=fosize)
    plt.xlabel("x (au)", fontsize=fosize)
    plt.ylabel("y (au)", fontsize=fosize)
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    
    plt.xticks(fontsize=fosize)
    plt.yticks(fontsize=fosize)
    plt.title("Individual Orbit at time t$\\approx$" + str(round(finalt/(oneyear), 3)) + " years, Target at (" + str(round(ibexpos[0]/au, 4)) + " au, " + str(round(ibexpos[1]/au, 4)) + " au, " + str(round(ibexpos[2]/au, 4)) + " au) \
        \n At target point v = (" + str(indxic/1000) + " km/s, " + str(indyic/1000) + " km/s, " + str(indzic/1000) + " km/s), Value of distribution function = " + str(round(psd, 13)) + " \
       \n Perihelion at $\\sim$ " + str(round(perihelion/au, 5)) + " au, Travel time from x = 100 au to target $\\approx$ " + str(round(ttime/oneyear, 3)) + " years", fontsize=10)
    plt.show()

    # extra code for plotting conserved values
    """f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    l1, = ax1.plot(t,Etrack, color='r')
    l2, = ax2.plot(t,Ltrack, color='b')
    l3, = ax3.plot(t,rtrack, color='y')
    ax1.legend((l1, l2, l3), ('Adjusted Specific Energy', 'Magnitude of Angular Momentum', 'Radius'), loc='upper left')
    f.subplots_adjust(hspace=.0)
    f.set_size_inches(8,4)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()"""
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
# code for tracking phase space at distance of x = 100 au away
if mode==3:
    farvx = np.array([])
    farvy = np.array([])
    fart = np.array([])
    maxwcolor = np.array([])
    vrmax = np.array([])
    vrmin = np.array([])
    backtraj = np.zeros((t.size, 6))
    for i in tqdm(range(vxstart.size)): # displays progress bars for both loops to measure progress
        for j in tqdm(range(vystart.size)):
            init = [xstart, ystart, zstart, vxstart[i], vystart[j], vzstart]
            # calculating trajectories for each initial condition in phase space given
            #print("\n" + str(np.sqrt((vxstart[i]/1000)**2 + (vystart[j]/1000)**2)))
            if np.sqrt((vxstart[i]/1000)**2 + (vystart[j]/1000)**2) < noncalcextent:
                #print("\n skipped")
                # skips calculating the trajectory if the initial velocity is within a certain distance in velocity space from the origin
                farvx = np.append(farvx, [vxstart[i]])
                farvy = np.append(farvy, [vystart[i]])
                fart = np.append(fart, [0])
                # sets the value of the NPSD to 0
                maxwcolor = np.append(maxwcolor, [0])
                continue
            thetarad = theta*np.pi/180
            fxea = 50*np.cos(thetarad) # distance of a point far away on the exclusion axis in the x direction
            fyea = 50*np.sin(thetarad) # distance of a point far away on the exclusion axis in the y direction
            origin = np.array([0,0,0])
            fea = np.array([fxea, fyea, 0])
            initialv = np.array([vxstart[i]/1000, vystart[j]/1000, vzstart/1000])
            if np.linalg.norm(np.cross(fea-origin, origin-initialv))/np.linalg.norm(fea-origin) < noncalcextent and np.abs(np.linalg.norm(fea-initialv)) < 50:
                # skips calculating the trajectory if it is too close to the axis of exclusion
                # checks distance to axis of exclusion, then checks if point is within 50 km/s of
                # 50 km/s from the origin along the axis of exclusion
                # since the effect of the axis of exclusion only goes one way from the origin
                #print(initialv)
                farvx = np.append(farvx, [vxstart[i]])
                farvy = np.append(farvy, [vystart[j]])
                fart = np.append(fart, [0])
                # sets the value of the NPSD to 0
                maxwcolor = np.append(maxwcolor, [0])
                continue
            backtraj[:,:] = odeint(Abs_dr_dt, init, t, args=(lya_abs_update,))
            btr = np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2)
            if any(btr <= .00465*au):
                # tells the code to not consider the trajectory if it at any point intersects the width of the sun
                farvx = np.append(farvx, [backtraj[0,3]])
                farvy = np.append(farvy, [backtraj[0,4]])
                fart = np.append(fart, [0])
                # sets the value of the NPSD to -1 to indicate the trajectory isn't viable, as a special flag to investigate later
                maxwcolor = np.append(maxwcolor, [-1])
                continue
            if np.all(btr[:] < refdist*au):
                # forgoes the following checks if the trajectory never passes through the plane at the reference distance upwind
                farvx = np.append(farvx, [backtraj[0,3]])
                farvy = np.append(farvy, [backtraj[0,4]])
                fart = np.append(fart, [0])
                # sets the value of the NPSD to 0 to indicate the trajectory isn't viable
                maxwcolor = np.append(maxwcolor, [0])
                continue
            for k in range(t.size - countoffset):
                if btr[k+countoffset] >= refdist*au and btr[k-1+countoffset] <= refdist*au:
                    # adjusting the indexing to avoid checking in the close regime
                    #kn = k+tclose.size
                    kn = k+countoffset
                    # radius in paper given to be 14 km/s
                    # only saving initial conditions corresponding to points that lie within this Maxwellian at reference distance
                    #if backtraj[k-1,3,(i)*vystart.size + (j)] <= -22000 and backtraj[k-1,3,(i)*vystart.size + (j)] >= -40000 and backtraj[k-1,4,(i)*vystart.size + (j)] <= 14000 and backtraj[k-1,4,(i)*vystart.size + (j)] >= -14000:
                    #if np.sqrt((backtraj[kn-1,3]+26000)**2 + (backtraj[kn-1,4])**2 + (backtraj[kn-1,5])**2) <= 27000:
                    # determining which distribution to use by calculating heliolongitude
                    endradxy = np.sqrt((sunpos[0]-backtraj[kn+1,0])**2 + (sunpos[1]-backtraj[kn+1,1])**2)
                    belowxaxis = backtraj[kn+1,1] < 0
                    ymask = belowxaxis*2*np.pi
                    longmask = -2*(belowxaxis-.5) # -1 if below x axis in xy plane, 1 if above
                    # if y < 0, longitude = 2pi-arccos(x/r), otherwise longitude = arccos(x/r)
                    endlongangle = ymask + np.arccos((backtraj[kn+1,0] - sunpos[0])/endradxy)*longmask
                    endlongangle = endlongangle*180/np.pi
                    # finding the initial value of the distribution function based on the interpolated distributions
                    endvelcoords = [backtraj[kn+1,5]/1000,backtraj[kn+1,4]/1000,backtraj[kn+1,3]/1000]
                    if endlongangle >= 0 and endlongangle <= 45:
                        anglepct = (endlongangle)/45
                        initpsd = interp1(endvelcoords)*(1 - anglepct) + interp2(endvelcoords)*anglepct
                    elif endlongangle > 45 and endlongangle <= 90:
                        anglepct = (endlongangle - 45)/45
                        initpsd = interp2(endvelcoords)*(1 - anglepct) + interp3(endvelcoords)*anglepct
                    elif endlongangle > 90 and endlongangle <= 135:
                        anglepct = (endlongangle - 90)/45
                        initpsd = interp3(endvelcoords)*(1 - anglepct) + interp4(endvelcoords)*anglepct
                    elif endlongangle > 135 and endlongangle <= 180:
                        anglepct = (endlongangle - 135)/45
                        initpsd = interp4(endvelcoords)*(1 - anglepct) + interp5(endvelcoords)*anglepct
                    elif endlongangle > 180 and endlongangle <= 225:
                        anglepct = (endlongangle - 180)/45
                        initpsd = interp5(endvelcoords)*(1 - anglepct) + interp6(endvelcoords)*anglepct
                    elif endlongangle > 225 and endlongangle <= 270:
                        anglepct = (endlongangle - 225)/45
                        initpsd = interp6(endvelcoords)*(1 - anglepct) + interp7(endvelcoords)*anglepct
                    elif endlongangle > 270 and endlongangle <= 315:
                        anglepct = (endlongangle - 270)/45
                        initpsd = interp7(endvelcoords)*(1 - anglepct) + interp8(endvelcoords)*anglepct
                    elif endlongangle > 315 and endlongangle <= 360:
                        anglepct = (endlongangle - 315)/45
                        initpsd = interp8(endvelcoords)*(1 - anglepct) + interp1(endvelcoords)*anglepct

                    # radial distance to the Sun at all points throughout the trajectory
                    currentrad = np.sqrt((sunpos[0]-backtraj[0:kn+1,0])**2 + (sunpos[1]-backtraj[0:kn+1,1])**2 + (sunpos[2]-backtraj[0:kn+1,2])**2)

                    # velocity squared at each point in time for the trajectory
                    currentvsq = np.square(backtraj[0:kn+1,3]) + np.square(backtraj[0:kn+1,4]) + np.square(backtraj[0:kn+1,5])
                    # thermal velocity (temperature taken from Federico's given temperature)
                    vth = np.sqrt(2 * 1.381*10**(-23) * 7500 / (1.672*10**(-27)))
                    # omega for the relative velocity
                    omegavs = np.abs(np.sqrt(currentvsq) - vsolarwindms)/vth
                    # calculating the current collision velocity of the particle and the SW
                    currentvrel = vth*(np.exp(-omegavs**2)/np.sqrt(np.pi) + (omegavs + 1/(2*omegavs))*scipy.special.erf(omegavs))
                    # calculating kinetic energy in keV at each point in time
                    currentKE = (.5 * mH * np.square(currentvrel)) * 6.242*10**(15)

                    # parameters for function to calculate charge exchange cross section
                    a1 = 4.049
                    a2 = 0.447
                    a3 = 60.5
                    # function for H-H+ charge exchange cross section, from Swaczyna et al. (2019)
                    # result is in cm^2
                    # https://ui.adsabs.harvard.edu/abs/2019AGUFMSH51C3344S/abstract
                    cxcrosssection = ((a1 - a2*np.log(currentKE))**2 * (1 - np.exp(-a3 / currentKE))**(4.5))*10**(-16)

                    #nsw = nsw1au * (r1/currentrad)**2 # assuming r^2 falloff for density (Sokol et al. 2019)

                    cxirate = nsw1au * currentvrel*100 * cxcrosssection

                    #print(cxirate)

                    # approximate time-averaged charge exchange photoionization rate from Sokol et al. 2019
                    #cxirate = 5*10**(-7)
                    # omega*t for each time point in the trajectory
                    omt = 2*np.pi/(3.47*10**(8))*t[0:kn+1]
                    # function for the photoionization rate at each point in time
                    PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
                    #PIrate2 = 1.21163*10**(-7) # time average of above
                    # calculating the component of the radial unit vector in each direction at each point in time
                    nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
                    nrvecy = (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
                    nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
                    # calculating the magnitude of v_r at each point in time
                    currentvr = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
                    # calculating maximum and minimum radial velocities throughout the whole trajectory
                    vrmax = np.append(vrmax, max(currentvr))
                    vrmin = np.append(vrmin, min(currentvr))
                    # integrand for the photoionization and charge exchange ionization losses
                    btintegrand = PIrate2/currentvr*(r1/currentrad)**2 + cxirate/currentvr*(r1/currentrad)**2

                    # calculation of heliographic latitude angle (polar angle)
                    #belowxy = backtraj[0:kn+1,2] < 0
                    #zmask = 2*(belowxy-.5)
                    #latangle = np.pi/2 + zmask*np.arcsin(np.abs(backtraj[0:kn+1,2] - sunpos[2])/currentrad[:])
                    
                    # calculation of attenuation factor based on heliographic latitude angle
                    #btintegrand = btintegrand*(.85*(np.sin(latangle))**2 + (np.cos(latangle))**2)
                    # calculation of attenuation factor
                    attfact = scipy.integrate.simps(btintegrand, currentrad)
                    # calculating the value of the phase space density after attenuation
                    psdval = np.exp(-np.abs(attfact))*initpsd
                    #if psdval > 10**(-11):
                    # retaining variables corresponding to vx, vy, t at the target point
                    farvx = np.append(farvx, [backtraj[0,3]])
                    farvy = np.append(farvy, [backtraj[0,4]])
                    fart = np.append(fart, [startt - t[kn-1]])
                    # calculating value of phase space density based on the value at the crossing of x = 100 au
                    maxwcolor = np.append(maxwcolor, [psdval])
                    #maxwcolor = np.append(maxwcolor, [initpsd])
                    #maxwcolor = np.append(maxwcolor, [np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2 + backtraj[kn-1,5]**2)/(10195)**2)])
                    break
                    #break
                if k == (t.size + countoffset) - 1:
                    print("fail")
                    farvx = np.append(farvx, [backtraj[0,3]])
                    farvy = np.append(farvy, [backtraj[0,4]])
                    fart = np.append(fart, [0])
                    # sets the value of the NPSD to 0 to indicate the trajectory isn't viable
                    maxwcolor = np.append(maxwcolor, [0])
                
                

print('Finished')

if mode==3:
    # DATA WRITING AND PLOTTING CODE

    # writing data to a file - need to change each time or it will overwrite previous file
    file = open(fpath, 'w')
    #file = open("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p1fluccosexprp_35pi36_0y_direct_cosexppi_tclose400.txt", "w")
    for i in range(farvx.size):
        # writes vx, vy, and attenuated NPSD value
        file.write(str(farvx[i]/1000) + ',' + str(farvy[i]/1000) + ',' + str(maxwcolor[i]) + '\n')
    file.close()

    # plotting a scatterplot of vx and vy at the target point, colored by the phase space density
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    fsize = 18
    #plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='rainbow') # linear scale
    plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-11))) # log scale, minimum value at vmin
    #plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm())
    plt.rcParams.update({'font.size': fsize})
    cb = plt.colorbar()
    #cb.set_label('Time at which orbit passes through 100 au (s)')
    #cb.set_label('Travel Time from 100 au to Point of Interest (s)')
    cb.set_label('Phase Space Density')
    #plt.xlim([-25, 25])
    #plt.ylim([-25, 25])
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
    plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
    #plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
    #plt.suptitle('VDF at target, at t $\\approx$ ' + str(round(finalt/(oneyear), 3)) + ' years, drawn from Maxwellian at 100 au centered on $v_x$ = -26 km/s')
    #plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
    #plt.title('Target at (' + str(round(ibexpos[0]/au, 3)) + ' au, ' + str(round(ibexpos[1]/au, 3)) + ' au), Time Resolution Close to Target = ' + str(tstepclose) + ' s')
    #plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
    plt.show()

    overallvrmax = max(vrmax)
    overallvrmin = min(vrmin)
    meanvrmax = np.mean(vrmax)
    meanvrmin = np.mean(vrmin)
    #print(vrmax)
    #print(vrmin)
    print("The maximum radial velocity out of all trajectories at all points in time is " + str(overallvrmax) + " m/s")
    print("The minimum radial velocity out of all trajectories at all points in time is " + str(overallvrmin) + " m/s")
    print("The mean of maximum radial velocities for all trajectories is: " + str(meanvrmax) + " m/s")
    print("The mean of minimum radial velocities for all trajectories is: " + str(meanvrmin) + " m/s")
    

    maxage = max(fart)

    print("The longest time a trajectory took to get to get to the target point from the boundary surface is " + str(maxage/oneyear) + " yrs")