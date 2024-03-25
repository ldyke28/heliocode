import numpy as np
from scipy.integrate import odeint
import scipy
from mpi4py import MPI
import os
import warnings
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # obtaining the rank of each node to split the workload later

# Filtering warnings (specifically odeint's ODEintWarning) as errors
# This allows us to catch these warnings as errors to later run trajectories that may
# have issues with the temporal resolution
warnings.filterwarnings("error", category=Warning)

# Opening the file to read in the input data
file = open("3Dinputfile.txt", "r")


# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
oneyear = 3.15545454545*10**7

# 120749800 for first force free
# 226250200 for second force free
finalt = float(file.readline().strip()) # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -50000000000 # time in the past to which the code should backtrace
tstep = 10000 # general time resolution
tstepclose = float(file.readline().strip()) # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 115

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
theta = float(file.readline().strip()) # angle with respect to upwind axis of target point
ibexrad = 1 # radial distance of target point from Sun
ibexx = ibexrad*np.cos(theta*np.pi/180)
ibexy = ibexrad*np.sin(theta*np.pi/180)
ibexpos = np.array([ibexx*au, ibexy*au, 0])
# implementation of target point that orbits around the sun
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])


# Initial Conditions for orbit starting at point of interest for backtracing
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]

# Set of initial conditions in velocity space
# vx/vy initial conditions are sampled on a grid with chosen resolution
vxstart = np.arange(float(file.readline().strip()), float(file.readline().strip()), float(file.readline().strip()))
vystart = np.arange(float(file.readline().strip()), float(file.readline().strip()), float(file.readline().strip()))
vzstart = np.arange(float(file.readline().strip()), float(file.readline().strip()), float(file.readline().strip()))

startt = finalt
lastt = initialt
tmid = startt - 200000000 # time at which we switch from high resolution to low resolution - a little more than half of a solar cycle
tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
t = np.concatenate((tclose, tfar))


datafilename1 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_001_H_RegAll.h5'
datafilename2 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_002_H_RegAll.h5'
datafilename3 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_003_H_RegAll.h5'
datafilename4 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_004_H_RegAll.h5'
datafilename5 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_005_H_RegAll.h5'
datafilename6 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_006_H_RegAll.h5'
datafilename7 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_007_H_RegAll.h5'
datafilename8 = 'VDF3D_HE013Ksw_PRB_Eclip256_R115_008_H_RegAll.h5'


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


def radPressure(t,x,y,z,v_r):
    # dummy function to model radiation pressure
    # takes the time as input and returns the radiation pressure function at that time
    #return (np.sin(2*np.pi*(t/347000000)))**2 + .5
    #return .7
    return 0

# extra radiation pressure functions for overlayed plots
def cosexprp(t,x,y,z,v_r):
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    return .75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))

def cosexpabs(t,x,y,z,vr):
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    r = np.sqrt(x**2 + y**2 + z**2)
    # calculating the latitudinal (polar) angle in 3D space
    if z >= 0:
        latangle = np.pi/2 - np.cos(z/r)
    else:
        latangle = np.pi/2 + np.cos(np.abs(z)/r)
    # calculating the longitudinal (azimuthal) angle in 3D space 
    if y > 0:
        longangle = np.tan(y/x)
    elif y == 0 and x > 0:
        longangle = 0
    elif y == 0 and x < 0:
        longangle = np.pi
    elif y < 0:
        longangle = 2*np.pi - np.tan(y/x)
    
    # calculating parameters from IKL et al. 2022 paper: https://ui.adsabs.harvard.edu/abs/2022ApJ...926...27K/abstract
    if r < 1:
        amp = 0
    else:
        amp = (r-1)/99
    mds = np.sign(x)*25
    disper = 10
    fittype = 2
    absval = amp*np.exp(-.5 * ((vr/1000 - mds)/disper)**fittype)
    return (.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))*(1 - absval)

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


def LyaRP(t,x,y,z,v_r):
    # a double (triple) Gaussian function to mimic the Lyman-alpha profile
    lyafunction = 1.25*np.exp(-(v_r/1000-55)**2/(2*25**2)) + 1.25*np.exp(-(v_r/1000+55)**2/(2*25**2)) + .55*np.exp(-(v_r/1000)**2/(2*25**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match divisor in first term of addfactor
    scalefactor = 1.8956
    # added value to ensure scaling is throughout solar cycle
    # matches total irradiance out to +-120 km/s
    #addfactor = ((1.3244/1.616) - 1)*(.75 + .243*np.e)*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    # matches total irradiance out to +-370 km/s
    addfactor = ((1.55363/1.8956) - 1)*(.75 + .243*np.e)*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    return scalefactor*(.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + addfactor)*lyafunction


def LyaRP2(t,x,y,z,v_r):
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

def LyaRP3(t,x,y,z,v_r):
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
    #Author: E. Samoylov, H. Mueller LISM Group (Adapted by L. Dyke for this code)
    #Date: 04.18.2023
    #Purpose: To confirm the graph that EQ14 produces in
    #         Kowalska-Leszczynska's 2018 paper
    #         Evolution of the Solar Lyα Line Profile during the Solar Cycle
    #https://iopscience.iop.org/article/10.3847/1538-4357/aa9f2a/pdf

    # note the above parameters are taken from Kowalska-Leszczynska et al. 2020
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


def LyaRP4(t,x,y,z,v_r):
    #Author: E. Samoylov, H. Mueller LISM Group (Adapted by L. Dyke for this code)
    #https://iopscience.iop.org/article/10.3847/1538-4357/aa9f2a/pdf
    # Revised version of the function from IKL et al. 2018 - time dependence introduced through parameters
    omegat = 2*np.pi/(3.47*10**(8))*t
    # added value to ensure scaling is correct throughout solar cycle
    # matches total irradiance out to +-120 km/s
    #addfactor = ((.973/.9089) - 1)*.85*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    # matches total irradiance out to +-370 km/s
    #addfactor = ((.97423/.91) - 1)*.85*1/(np.e + 1/np.e)*(1/np.e + np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))
    # time dependent portion of the radiation pressure force function
    #tdependence = .85 - np.e/(np.e + 1/np.e)*.33 + .33/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + addfactor
    tdependence = .95 + .5/(np.e**2 + 1) + .5/(np.e + 1/np.e)*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match divisor in first term of addfactor
    scalefactor = .555
    
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
    F_R = A_R / (del_R * np.sqrt(2 * np.pi)) *np.exp(-(np.square((v_r/1000) - (m_K + dm))) / (2*(del_R ** 2)))
    F_bkg = np.add(a_bkg*(v_r/1000)*0.000001,b_bkg)
    F_K = A_K * np.power(1 + np.square((v_r/1000) - m_K) / (2 * K * ((del_K) ** 2)), -K - 1)

    r = np.sqrt(x**2 + y**2 + z**2)
    # calculating the latitudinal (polar) angle in 3D space
    if z >= 0:
        latangle = np.pi/2 - np.arcsin(z/r)
    else:
        latangle = np.pi/2 + np.arcsin(np.abs(z)/r)

    alya = .8
    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    return scalefactor*(F_K-F_R+F_bkg)/(r_E/(r2**2))*np.sqrt(alya*np.sin(latangle)**2 + np.cos(latangle)**2)


def lya_abs(t,x,y,z,vr):
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
        np.e**(-(longangled - 90)**2/50**2 - (r - 31)**2/15**2)*(1 + \
        scipy.special.erf(alpha*(r/au)/np.sqrt(2)))*(1 - np.e**(-(r/au)/4)))*1/.966
    
    # mean Doppler shift
    mds = 20*np.sin(longangle)*np.cos((latangled-10)*np.pi/180)
    # dispersion (width of the peak)
    disper = -.0006947*(r/au)**2 + .1745*(r/au) + 5.402 + \
        1.2*np.e**(-(longangled - 275)**2/50**2 - ((r/au) - 80)**2/60**2) + \
        3*np.e**(-(longangled - 90)**2/50**2 - ((r/au))**2/5**2) + \
        1*np.e**(-(longangled - 100)**2/50**2 - ((r/au) - 25)**2/200**2) + \
        .35*np.cos(((latangled + 15)*np.pi/180)*2)
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
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match divisor in first term of addfactor
    scalefactor = .555
    
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
    return scalefactor*(F_K-F_R+F_bkg)/(r_E/(r2**2))*(1 - absval)


# odeint documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html


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
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-rp(t,v_r))
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-rp(t,v_r))
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-rp(t,v_r))
    return [dx0, dx1, dx2, dx3, dx4, dx5]


def Var_dr_dt(x,t,rp):
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

# identify the total number of processes
nprocs = comm.Get_size()

# creating a shared array with the size of the maximum possible number of points that could exist
size = vxstart.size * vystart.size * vzstart.size
itemsize = MPI.FLOAT.Get_size()
if rank == 0:
    nbytes = 5*size*itemsize # saving five variables - need to multiply by 5 here
else:
    nbytes = 0

# creating a shared block on rank 0 and a window to it
win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

# creating arrays whose data points to shared memory
buf, itemsize = win.Shared_query(0)
assert itemsize == MPI.FLOAT.Get_size()
data = np.ndarray(buffer=buf, dtype='f', shape=(size,5))

# Initializing an array to determine where to slice initial conditions for vx
bounds = np.zeros(nprocs, dtype=int)

# Setting the location of these bounds by dividing the total number of initial vx conditions as evenly as possible
for q in range(nprocs-1):
    bounds[q+1] = int(np.floor(vxstart.size/(nprocs-1)*(q+1)))

# To sum loss counts at the end, we need to have them as numpy objects
sunlosscount = np.zeros(1)
sunlosscounttotal = np.zeros(1)
dirlosscount = np.zeros(1)
dirlosscounttotal = np.zeros(1)

# Initializing array to collect problematic points to test afterward
lostpoints = np.array([0,0,0])
for m in range(nprocs-1):
    if rank == m+1:
        vxstartn = vxstart[bounds[m]:(bounds[m+1]+1)]
        for i in range(vxstartn.size): # displays progress bars for both loops to measure progress
            restartfile = open('restart%s' % rank, 'a')
            for j in range(vystart.size):
                for l in range(vzstart.size):
                    init = [xstart, ystart, zstart, vxstartn[i], vystart[j], vzstart[l]]
                    try:
                        # Main code in try block
                        # If an ODEintWarning is raised, point will be set aside for testing later on
                        # calculating trajectories for each initial condition in phase space given
                        backtraj = odeint(Var_dr_dt, init, t, args=(LyaRP4,))
                        btr = np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2)
                        if any(btr <= .00465*au):
                            # tells the code to not consider the trajectory if it at any point intersects the width of the sun
                            sunlosscount[0] += 1
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,0] = vxstartn[i]
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,1] = vystart[j]
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,2] = vzstart[l]
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,3] = 0
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,4] = 0
                            continue
                        if all(btr < refdist*au):
                            # forgoes the following checks if the trajectory never passes through x = 100 au
                            dirlosscount[0] += 1
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,0] = vxstartn[i]
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,1] = vystart[j]
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,2] = vzstart[l]
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,3] = 0
                            data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,4] = 0
                            continue
                        for k in range(t.size - tclose.size):
                            if btr[k+tclose.size] >= refdist*au and btr[k-1+tclose.size] <= refdist*au:
                                # adjusting the indexing to avoid checking in the close regime
                                kn = k+tclose.size
                                # radius in paper given to be 14 km/s
                                # only saving initial conditions corresponding to points that lie within this Maxwellian at x = 100 au
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

                                omt = 2*np.pi/(3.47*10**(8))*t[0:kn+1]
                                # function for the charge exchange ionization rate
                                cxirate = 5*10**(-7)
                                # function for the photoionization rate at each point in time
                                PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
                                r1 = 1*au # reference radius
                                # radial distance to the Sun at all points throughout the trajectory
                                currentrad = np.sqrt((sunpos[0]-backtraj[0:kn+1,0])**2 + (sunpos[1]-backtraj[0:kn+1,1])**2 + (sunpos[2]-backtraj[0:kn+1,2])**2)
                                # calculating the component of the radial unit vector in each direction at each point in time
                                nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
                                nrvecy= (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
                                nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
                                # calculating the magnitude of v_r at each point in time
                                currentvr = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
                                # integrand for the photoionization losses
                                btintegrand = PIrate2/currentvr*(r1/currentrad)**2 + + cxirate/currentvr*(r1/currentrad)**2
                                # calculation of heliographic latitude angle (polar angle)
                                belowxy = backtraj[0:kn+1,2] < 0
                                zmask = 2*(belowxy-.5)
                                latangle = np.pi/2 + zmask*np.arcsin(np.abs(backtraj[0:kn+1,2] - sunpos[2])/currentrad[:])
                                
                                # calculation of attenuation factor based on heliographic latitude angle
                                btintegrand = btintegrand*(.85*(np.sin(latangle))**2 + (np.cos(latangle))**2)

                                # calculation of attenuation factor
                                attfact = scipy.integrate.simps(btintegrand, currentrad)
                                # calculating value of phase space density based on the value at the crossing of x = 100 au
                                attenval = np.exp(-np.abs(attfact))*initpsd[0]
                                
                                # storing relevant values in a shared array
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,0] = vxstartn[i]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,1] = vystart[j]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,2] = vzstart[l]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,3] = startt - t[kn-1]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,4] = np.exp(-np.abs(attfact))*initpsd[0]
                                restartfile.write(str(vxstartn[i]/1000) + ',' + str(vystart[j]/1000) + ',' + str(vzstart[l]/1000) + ',' + str(attenval) + '\n')
                                break
                                #break
                            if k == (t.size - tclose.size) - 1:
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,0] = vxstartn[i]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,1] = vystart[j]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,2] = vzstart[l]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,3] = 0
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,4] = 0
                    except Warning:
                        # Collects the points that seem to cause issues to be ran again with different temporal resolution
                        #lostpoints = np.vstack([lostpoints, [vxstartn[i], vystart[j], vzstart[l]]])
                        lostpoints = np.append(lostpoints, [vxstartn[i], vystart[j], vzstart[l]])
                        #file2.write(str(vxstartn[i]) + ',' + str(vystart[j]) + ',' + str(vzstart[l]) + '\n')
            restartfile.close()           
                    
        break

# Forces processes to wait until all have finished before moving on
comm.Barrier()

print('Finished')

# Resets warnings to be treated as warnings once more
warnings.filterwarnings("default", category=Warning)

# Determining how many array elements are in each array of problem points on each process
sendcounts = np.array(comm.gather(len(lostpoints), 0))
if rank == 0:
    # Creates an array on the master node to collect all of the points
    recvbuf = np.empty(sum(sendcounts), dtype=float)
else:
    recvbuf = None

# Gathers the problem points into a single array into a single array on the master node
comm.Gatherv(sendbuf=lostpoints, recvbuf=(recvbuf, sendcounts), root=0)

# Sums up all of the loss counts into a single count on the master node
comm.Reduce(sunlosscount, sunlosscounttotal, op=MPI.SUM, root=0)
comm.Reduce(dirlosscount, dirlosscounttotal, op=MPI.SUM, root=0)

# writing data to a file - need to change each time or it will overwrite previous file
if rank == 0:
    # masking the points in the completed trajectories that are irrelevant
    data = data[~np.all(data == 0, axis=1)]
    fname = file.readline().strip()
    dfile = open(fname, 'w')
    for i in range(np.size(data, 0)):
        # writing relevant data points to a file
        dfile.write(str(data[i,0]/1000) + ',' + str(data[i,1]/1000) + ',' + str(data[i,2]/1000) + ',' + str(data[i,4]) + '\n')
    dfile.close()

    # writing loss counts to a file
    lossfile = open('losses_' + fname, 'w')
    lossfile.write('Particle losses to the sun: ' + str(sunlosscounttotal[0]) + '\n' + 'Trajectories not intersecting x = 100 au plane: ' + str(dirlosscounttotal[0]))
    lossfile.close()

    # writing problem points to a file
    file2 = open("lostpoints_" + fname, "w")
    for i in range(int(np.floor(np.size(recvbuf, 0)/3))):
        file2.write(str(recvbuf[3*i]) + ',' + str(recvbuf[3*i+1]) + ',' + str(recvbuf[3*i+2]) + '\n')
    file2.close()
    print('All done!')

comm.Barrier()
file.close()
print("Completed.")